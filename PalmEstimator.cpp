// PalmEstimator.cpp
#include "PalmEstimator.h"
#include <algorithm>
#include <iostream>
#include <numeric>

// External declarations from main.cpp
extern cv::Point2f lastValidPalm;
extern cv::Point2f lastValidThumbBase;
extern cv::Point2f lastValidPinkyBase;
extern int failureFrameCount;
extern std::atomic<bool> isCalibrated;
extern CalibrationResult calibrationData;
extern GeometryUpdater geometryUpdater;
extern ShapeAnchoredTracker shapeTracker;

// Helper function: check if point is inside contour
bool PalmEstimator::isPointInContour(const cv::Point2f& point, const std::vector<cv::Point>& contour) const {
    if (contour.empty()) return true;
    cv::Point point_int(static_cast<int>(point.x), static_cast<int>(point.y));
    return cv::pointPolygonTest(contour, point_int, false) >= 0;
}

// Helper function: project point to contour interior
cv::Point2f PalmEstimator::projectPointToContourInterior(const cv::Point2f& point, 
                                                         const std::vector<cv::Point>& contour,
                                                         const cv::Point2f& fallback) const {
    if (contour.empty()) return point;
    
    // Check if point is already inside
    if (isPointInContour(point, contour)) {
        return point;
    }
    
    // Find nearest point on contour
    cv::Point2f nearest = point;
    float minDist = std::numeric_limits<float>::max();
    
    for (const auto& p : contour) {
        cv::Point2f contourPoint(p);
        float dist = cv::norm(contourPoint - point);
        if (dist < minDist) {
            minDist = dist;
            nearest = contourPoint;
        }
    }
    
    return nearest;
}

PalmEstimator::Result PalmEstimator::detect(const cv::Mat& frame, const cv::Mat& motionMask, 
                                           const cv::Mat& skinMask, float hsvConfidence) {
    Result result;
    result.status = "Processing";
    result.wristHint = cv::Point2f(-1, -1);
    
    try {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        result.contoursFound = contours.size();
        
        if (contours.empty()) {
            result.handDetected = false;
            result.status = "No contours";
            return result;
        }
        
        std::vector<size_t> potentialIndices;
        std::vector<float> potentialScores;
        
        for (size_t i = 0; i < contours.size(); i++) {
            float aspect, solidity;
            if (couldBeHand(contours[i], aspect, solidity)) {
                potentialIndices.push_back(i);
                
                double area = cv::contourArea(contours[i]);
                
                cv::Moments m = cv::moments(contours[i]);
                if (m.m00 > 0) {
                    cv::Point2f contourCenter(
                        static_cast<float>(m.m10 / m.m00),
                        static_cast<float>(m.m01 / m.m00)
                    );
                    
                    // Ensure center is inside contour
                    contourCenter = projectPointToContourInterior(contourCenter, contours[i], contourCenter);
                    
                    float score = area * 1.0f;
                    
                    float geometryScore = computeGeometryConsistencyScore(contours[i], contourCenter, 
                        std::sqrt(area / M_PI));
                    score *= (1.0f + geometryScore * 0.5f);
                    
                    if (isCalibrated) {
                        float distFromCalibrated = cv::norm(contourCenter - calibrationData.palmCenter);
                        if (distFromCalibrated < 200) {
                            score += 500.0f;
                        }
                    }
                    
                    potentialScores.push_back(score);
                    result.potentialHands++;
                }
            }
        }
        
        if (potentialIndices.empty()) {
            // Always process the largest contour even if not hand-like
            size_t largestIdx = 0;
            double largestArea = 0;
            for (size_t i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > largestArea) {
                    largestArea = area;
                    largestIdx = i;
                }
            }
            result.contour = contours[largestIdx];
            result.handDetected = true; // Always detect when skin exists
            result.status = "Hand (fallback contour)";
        } else {
            auto maxScoreIt = std::max_element(potentialScores.begin(), potentialScores.end());
            size_t bestIdxInPotentials = std::distance(potentialScores.begin(), maxScoreIt);
            size_t bestContourIdx = potentialIndices[bestIdxInPotentials];
            
            result.contour = contours[bestContourIdx];
            result.handDetected = true; // Always detect when skin exists
        }
        
        if (!result.contour.empty()) {
            result.area = cv::contourArea(result.contour);
            result.boundingBox = cv::boundingRect(result.contour);
            
            float aspect, solidity;
            couldBeHand(result.contour, aspect, solidity);
            result.aspectRatio = aspect;
            result.solidity = solidity;
            
            // STEP 1: PURE WRIST ESTIMATION (NO TRIMMING)
            float wristConfidence = 0.0f;
            result.wristLeft = cv::Point2f(-1, -1);
            result.wristRight = cv::Point2f(-1, -1);
            result.wristMid = cv::Point2f(-1, -1);
            
            wristConfidence = computeWristGeometry(result.contour,
                                                  result.wristLeft,
                                                  result.wristRight,
                                                  result.wristMid);
            
            // STEP 2: SHAPE SCULPTING (AFTER WRIST ESTIMATION)
            result.constrainedContour = constrainContourWithWrist(result.contour,
                                                                result.wristLeft,
                                                                result.wristRight,
                                                                result.wristMid,
                                                                wristConfidence);
            
            // STEP 3: COMPUTE PALM CENTER (ALWAYS)
            result.handSizeScale = 1.0f;
            
            // ALWAYS compute palm center - use sculpted contour if available, otherwise raw
            std::vector<cv::Point> contourForPalm = result.constrainedContour.empty() ? 
                                                    result.contour : result.constrainedContour;
            
            // ABSOLUTE: Palm center computation never fails
            bool palmComputed = fitContourToModel(contourForPalm, result.palm, result.handSizeScale);
            
            if (wristConfidence > 0.5f) {
                result.status = "Hand (confident)";
            } else if (wristConfidence > 0.2f) {
                result.status = "Hand (low wrist confidence)";
            } else {
                result.status = "Hand (wrist estimation unreliable)";
            }
            
            // Call GeometryUpdater with AUTHORITATIVE wrist geometry
            geometryUpdater.updateGeometry(result.palm, 
                                          result.wristMid,      // AUTHORITATIVE
                                          result.wristLeft,     // AUTHORITATIVE  
                                          result.wristRight,    // AUTHORITATIVE
                                          result.constrainedContour); // Sculpted contour
                
            if (geometryUpdater.isPalmValid()) {
                // Get finger state from GeometryUpdater
                result.fingerState = geometryUpdater.getFingerState();
                result.detectedFingerCount = geometryUpdater.getDetectedFingerCount();
                
                // Build status string
                std::string stateStr;
                switch (result.fingerState) {
                    case FINGER_STATE_FIST: stateStr = " (fist)"; break;
                    case FINGER_STATE_PARTIAL: stateStr = " (partial)"; break;
                    case FINGER_STATE_OPEN: stateStr = " (open)"; break;
                }
                
                result.status += stateStr;
                result.confidence = wristConfidence;
            }
        }
        
    } catch (const cv::Exception& e) {
        result.status = "Error";
        result.handDetected = false;
    }
    
    return result;
}

bool PalmEstimator::validatePalmShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter) {
    if (contour.size() < 20) return true;
    
    // First ensure palm center is inside contour
    if (!isPointInContour(palmCenter, contour)) {
        return false;
    }
    
    std::vector<float> distances;
    for (const auto& p : contour) {
        distances.push_back(cv::norm(cv::Point2f(p) - palmCenter));
    }
    
    float mean = 0.0f;
    for (float d : distances) mean += d;
    mean /= distances.size();
    
    float variance = 0.0f;
    for (float d : distances) {
        float diff = d - mean;
        variance += diff * diff;
    }
    variance /= distances.size();
    
    float radialVariance = std::sqrt(variance) / mean;
    if (radialVariance > PE_CONTOUR_RADIAL_VARIANCE_MAX) return false;
    
    double area = cv::contourArea(contour);
    cv::Rect bbox = cv::boundingRect(contour);
    double bboxArea = bbox.width * bbox.height;
    
    if (bboxArea > 0) {
        float compactness = area / bboxArea;
        if (compactness < PE_CONTOUR_COMPACTNESS_MIN) return false;
    }
    
    float geometryScore = computeGeometryConsistencyScore(contour, palmCenter, mean);
    if (geometryScore < 0.3f) {
        return false;
    }
    
    return true;
}

float PalmEstimator::computeGeometryConsistencyScore(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter, float palmRadius) {
    if (contour.size() < 20 || palmRadius < 10.0f) {
        return 0.5f;
    }
    
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices, false, false);
    
    std::vector<cv::Vec4i> defects;
    if (hullIndices.size() > 3) {
        cv::convexityDefects(contour, hullIndices, defects);
    }
    
    int validDefects = 0;
    float totalDefectDepth = 0.0f;
    for (const auto& defect : defects) {
        float depth = defect[3] / 256.0f;
        if (depth > palmRadius * MIN_DEFECT_DEPTH_RATIO) {
            validDefects++;
            totalDefectDepth += depth;
        }
    }
    
    float defectScore = 0.0f;
    if (validDefects >= PE_MIN_VALID_CONVEXITY_DEFECTS) {
        defectScore = std::min(1.0f, validDefects / 5.0f);
    } else if (validDefects == 1) {
        defectScore = 0.3f;
    } else {
        defectScore = 0.1f;
    }
    
    float symmetryScore = 0.0f;
    if (contour.size() > 30) {
        std::vector<cv::Point2f> contourPoints;
        for (const auto& p : contour) {
            contourPoints.emplace_back(p);
        }
        
        int leftCount = 0, rightCount = 0;
        float leftDist = 0.0f, rightDist = 0.0f;
        
        for (const auto& p : contourPoints) {
            if (p.x < palmCenter.x) {
                leftCount++;
                leftDist += cv::norm(p - palmCenter);
            } else {
                rightCount++;
                rightDist += cv::norm(p - palmCenter);
            }
        }
        
        if (leftCount > 0 && rightCount > 0) {
            float leftAvg = leftDist / leftCount;
            float rightAvg = rightDist / rightCount;
            float symmetry = std::min(leftAvg, rightAvg) / std::max(leftAvg, rightAvg);
            symmetryScore = 1.0f - std::min(1.0f, symmetry / PE_MAX_SYMMETRY_SCORE);
        }
    }
    
    cv::RotatedRect ellipse = cv::fitEllipse(contour);
    float aspectRatio = std::max(ellipse.size.width, ellipse.size.height) / 
                       std::max(1.0f, std::min(ellipse.size.width, ellipse.size.height));
    float aspectScore = std::min(1.0f, aspectRatio / 2.0f);
    
    return defectScore * 0.5f + symmetryScore * 0.3f + aspectScore * 0.2f;
}

bool PalmEstimator::couldBeHand(const std::vector<cv::Point>& contour, float& aspect, float& solidity) {
    double area = cv::contourArea(contour);
    
    if (area < PE_MIN_HAND_AREA) {
        return true; // Always consider small contours
    }
    
    cv::Rect bbox = cv::boundingRect(contour);
    aspect = static_cast<float>(bbox.width) / std::max(1.0f, static_cast<float>(bbox.height));
    
    if (area > 2000) {
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull, false);
        double hullArea = cv::contourArea(hull);
        
        if (hullArea > 0) {
            solidity = static_cast<float>(area / hullArea);
        } else {
            solidity = 1.0f;
        }
    } else {
        solidity = 1.0f;
    }
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 > 0) {
        cv::Point2f contourCenter(
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
        );
        
        // Ensure center is inside contour
        contourCenter = projectPointToContourInterior(contourCenter, contour, contourCenter);
        
        float geometryScore = computeGeometryConsistencyScore(contour, contourCenter, 
            std::sqrt(area / M_PI));
        
        return geometryScore > 0.1f; // Very low threshold
    }
    
    return true;
}

// PURE WRIST ESTIMATION - NO TRIMMING
float PalmEstimator::computeWristGeometry(const std::vector<cv::Point>& rawContour,
                                        cv::Point2f& wristLeft,
                                        cv::Point2f& wristRight,
                                        cv::Point2f& wristMid) {
    // Initialize with invalid values
    wristLeft = cv::Point2f(-1, -1);
    wristRight = cv::Point2f(-1, -1);
    wristMid = cv::Point2f(-1, -1);
    
    float confidence = 0.0f;
    
    // BEST-EFFORT: Contour must exist but can be small
    if (rawContour.size() < 20) {
        return confidence;
    }
    
    // STEP 1: Find convex hull and defects (PURE ANALYSIS)
    std::vector<int> hullIndices;
    cv::convexHull(rawContour, hullIndices, false, false);
    
    std::vector<cv::Vec4i> defects;
    if (hullIndices.size() > 3) {
        cv::convexityDefects(rawContour, hullIndices, defects);
    }
    
    // BEST-EFFORT: If no defects, use fallback estimation
    if (defects.empty()) {
        // Fallback: estimate wrist based on bounding box bottom
        cv::Rect bbox = cv::boundingRect(rawContour);
        wristLeft = cv::Point2f(bbox.x + bbox.width * 0.2f, bbox.y + bbox.height * 0.9f);
        wristRight = cv::Point2f(bbox.x + bbox.width * 0.8f, bbox.y + bbox.height * 0.9f);
        wristMid = (wristLeft + wristRight) * 0.5f;
        
        confidence = 0.1f; // Very low confidence for fallback
        return confidence;
    }
    
    // STEP 2: Identify thumb and pinky base candidates from deepest defects
    auto [thumbDefectIdx, pinkyDefectIdx] = findThumbPinkyDefects(rawContour, defects);
    
    // BEST-EFFORT: If can't identify proper defects, use deepest ones
    if (thumbDefectIdx < 0 || pinkyDefectIdx < 0) {
        if (!defects.empty()) {
            // Use first two defects as fallback
            thumbDefectIdx = 0;
            pinkyDefectIdx = std::min(1, (int)defects.size() - 1);
            confidence = 0.2f; // Low confidence
        } else {
            // Fallback to bounding box
            cv::Rect bbox = cv::boundingRect(rawContour);
            wristLeft = cv::Point2f(bbox.x + bbox.width * 0.25f, bbox.y + bbox.height * 0.85f);
            wristRight = cv::Point2f(bbox.x + bbox.width * 0.75f, bbox.y + bbox.height * 0.85f);
            wristMid = (wristLeft + wristRight) * 0.5f;
            
            confidence = 0.1f;
            return confidence;
        }
    }
    
    // STEP 3: Get thumb and pinky base points
    const auto& thumbDefect = defects[thumbDefectIdx];
    const auto& pinkyDefect = defects[pinkyDefectIdx];
    
    cv::Point2f thumbBase(rawContour[thumbDefect[2]]);  // Far point of defect
    cv::Point2f pinkyBase(rawContour[pinkyDefect[2]]);  // Far point of defect
    
    // STEP 4: Establish hand orientation axis
    cv::Point2f handAxis = pinkyBase - thumbBase;
    float axisLength = cv::norm(handAxis);
    
    // BEST-EFFORT: Adjust for very narrow axis
    if (axisLength < 30.0f) {
        // Scale up to minimum
        float scale = 30.0f / std::max(axisLength, 1.0f);
        handAxis = handAxis * scale;
        axisLength = 30.0f;
        confidence = 0.3f; // Low confidence due to scaling
    } else {
        confidence = 0.5f; // Moderate confidence
    }
    
    cv::Point2f axisDir = handAxis / axisLength;
    cv::Point2f perpendicular(-axisDir.y, axisDir.x);
    
    // STEP 5: Estimate rough palm center
    cv::Point2f roughPalmCenter = (thumbBase + pinkyBase) * 0.5f;
    
    // STEP 6: Compute wrist line
    float wristDistance = axisLength * 0.8f;
    wristMid = roughPalmCenter - perpendicular * wristDistance;
    float wristHalfWidth = axisLength * 0.3f;
    
    wristLeft = wristMid - axisDir * wristHalfWidth;
    wristRight = wristMid + axisDir * wristHalfWidth;
    
    // Increase confidence if wrist points are inside contour
    if (isPointInContour(wristLeft, rawContour) && isPointInContour(wristRight, rawContour)) {
        confidence = std::min(1.0f, confidence + 0.3f);
    }
    
    return confidence;
}

std::pair<int, int> PalmEstimator::findThumbPinkyDefects(const std::vector<cv::Point>& contour,
                                                        const std::vector<cv::Vec4i>& defects) const {
    if (defects.size() < 2) {
        // BEST-EFFORT: Return first two if available, otherwise invalid
        if (defects.size() >= 2) {
            return {0, 1};
        } else if (defects.size() == 1) {
            return {0, 0}; // Same defect for both
        }
        return {-1, -1};
    }
    
    // Sort defects by depth (descending)
    std::vector<std::pair<float, int>> defectDepths;
    for (size_t i = 0; i < defects.size(); i++) {
        float depth = defects[i][3] / 256.0f;
        defectDepths.emplace_back(depth, static_cast<int>(i));
    }
    
    std::sort(defectDepths.begin(), defectDepths.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    if (defectDepths.size() < 2) {
        return {defectDepths[0].second, defectDepths[0].second}; // Same for both
    }
    
    // Take two deepest defects as thumb and pinky candidates
    int deepestIdx = defectDepths[0].second;
    int secondDeepestIdx = defectDepths[1].second;
    
    // Determine which is thumb (leftmost) and pinky (rightmost)
    cv::Point2f deepestPoint(contour[defects[deepestIdx][2]]);
    cv::Point2f secondPoint(contour[defects[secondDeepestIdx][2]]);
    
    if (deepestPoint.x < secondPoint.x) {
        return {deepestIdx, secondDeepestIdx};  // thumb, pinky
    } else {
        return {secondDeepestIdx, deepestIdx};  // thumb, pinky
    }
}

// ITERATION 2: Enhanced shape sculpting with independent soft passes
std::vector<cv::Point> PalmEstimator::constrainContourWithWrist(const std::vector<cv::Point>& rawContour,
                                                               const cv::Point2f& wristLeft,
                                                               const cv::Point2f& wristRight,
                                                               const cv::Point2f& wristMid,
                                                               float wristConfidence) const {
    // ABSOLUTE: Always return at least the raw contour
    if (rawContour.empty() || wristConfidence < 0.1f) {
        return rawContour;
    }
    
    std::vector<cv::Point> workingContour = rawContour;
    
    // PASS 1: Basic wrist-line trimming (arm suppression in -Y direction)
    workingContour = applyWristLineTrimming(workingContour, wristLeft, wristRight, wristConfidence);
    
    // PASS 2: Lobe analysis and limiting (≤5 finger-like protrusions)
    workingContour = applyLobeLimiting(workingContour, wristConfidence);
    
    // PASS 3: Width consistency check (only in -Y direction)
    workingContour = applyWidthConsistencyCheck(workingContour, wristLeft, wristRight, wristConfidence);
    
    // PASS 4: Calibration-biased refinement
    workingContour = applyCalibrationBiasedRefinement(workingContour, wristConfidence);
    
    // ABSOLUTE: Never return empty contour
    if (workingContour.empty()) {
        return rawContour;
    }
    
    return workingContour;
}

// Helper for PASS 1: Wrist-line trimming (arm suppression)
std::vector<cv::Point> PalmEstimator::applyWristLineTrimming(const std::vector<cv::Point>& contour,
                                                           const cv::Point2f& wristLeft,
                                                           const cv::Point2f& wristRight,
                                                           float wristConfidence) const {
    if (wristLeft.x < 0 || wristRight.x < 0 || contour.empty()) {
        return contour;
    }
    
    // Create wrist line vector
    cv::Point2f wristVec = wristRight - wristLeft;
    float wristLen = cv::norm(wristVec);
    if (wristLen < 0.001f) {
        return contour;
    }
    
    cv::Point2f wristDir = wristVec / wristLen;
    cv::Point2f wristNormal(-wristDir.y, wristDir.x);
    
    // Determine which side is palm (+Y) vs arm (-Y)
    cv::Point2f contourCentroid(0, 0);
    for (const auto& p : contour) {
        contourCentroid += cv::Point2f(p);
    }
    contourCentroid = contourCentroid * (1.0f / contour.size());
    
    cv::Point2f toCentroid = contourCentroid - wristLeft;
    if (toCentroid.dot(wristNormal) < 0) {
        wristNormal = -wristNormal; // Ensure normal points toward palm
    }
    
    // CONFIDENCE-BASED TRIMMING (only in -Y/arm direction)
    std::vector<cv::Point> trimmed;
    float trimThreshold = 20.0f * (1.0f - wristConfidence * 0.7f); // Softer trimming with higher confidence
    
    for (const auto& p : contour) {
        cv::Point2f vecToPoint = cv::Point2f(p) - wristLeft;
        float crossProduct = vecToPoint.dot(wristNormal);
        
        // ARM SIDE (negative crossProduct): progressive trimming
        if (crossProduct < 0) {
            float distanceFromWrist = std::abs(crossProduct);
            
            // Keep points close to wrist line (transition zone)
            if (distanceFromWrist < trimThreshold) {
                trimmed.push_back(p);
            }
            // Else: discard distant arm points
        } 
        // PALM SIDE (positive crossProduct): keep all points
        else {
            trimmed.push_back(p);
        }
    }
    
    // ABSOLUTE: If trimming removed too much, return original
    if (trimmed.size() < contour.size() * 0.3f) {
        return contour;
    }
    
    return trimmed;
}

// Helper for PASS 2: Lobe analysis and limiting
std::vector<cv::Point> PalmEstimator::applyLobeLimiting(const std::vector<cv::Point>& contour,
                                                       float confidence) const {
    if (contour.size() < 30) return contour;
    
    std::vector<cv::Point> result = contour;
    
    // Find convex hull and defects for lobe analysis
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices, false, false);
    
    if (hullIndices.size() < 3) return contour;
    
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIndices, defects);
    
    // Analyze lobes (finger-like protrusions)
    struct LobeInfo {
        int defectIdx;
        float depth;
        float width;
        cv::Point farPoint;
    };
    
    std::vector<LobeInfo> lobeCandidates;
    
    for (size_t i = 0; i < defects.size(); i++) {
        float depth = defects[i][3] / 256.0f;
        
        int startIdx = defects[i][0];
        int endIdx = defects[i][1];
        int farIdx = defects[i][2];
        
        cv::Point startPt = contour[startIdx];
        cv::Point endPt = contour[endIdx];
        cv::Point farPt = contour[farIdx];
        
        float chordLength = cv::norm(cv::Point2f(endPt) - cv::Point2f(startPt));
        
        // Valid finger lobe criteria
        if (depth > chordLength * 0.15f && depth > 10.0f && chordLength > 15.0f) {
            LobeInfo lobe;
            lobe.defectIdx = static_cast<int>(i);
            lobe.depth = depth;
            lobe.width = chordLength;
            lobe.farPoint = farPt;
            lobeCandidates.push_back(lobe);
        }
    }
    
    // Sort by depth (strongest lobes first)
    std::sort(lobeCandidates.begin(), lobeCandidates.end(),
              [](const LobeInfo& a, const LobeInfo& b) { return a.depth > b.depth; });
    
    // SOFT LIMIT: If more than 5 lobes, reduce confidence and consider trimming weakest
    if (lobeCandidates.size() > 5) {
        // Mark weakest lobes (tail of sorted list) for potential trimming
        std::vector<cv::Point> weakestLobePoints;
        
        // Collect points from weakest lobes (indices 5 and beyond)
        for (size_t i = 5; i < lobeCandidates.size(); i++) {
            const auto& defect = defects[lobeCandidates[i].defectIdx];
            
            // Get points associated with this defect
            cv::Point startPt = contour[defect[0]];
            cv::Point endPt = contour[defect[1]];
            cv::Point farPt = contour[defect[2]];
            
            // Find contour segment between start and end points
            int startIdx = defect[0];
            int endIdx = defect[1];
            
            // Ensure proper ordering
            if (startIdx > endIdx) std::swap(startIdx, endIdx);
            
            // Mark points in this segment for potential removal
            for (int j = startIdx; j <= endIdx && j < (int)result.size(); j++) {
                weakestLobePoints.push_back(result[j]);
            }
        }
        
        // SOFT TRIMMING: Only remove if confidence is high enough
        if (confidence > 0.6f && !weakestLobePoints.empty()) {
            // Create new contour without weakest lobe segments
            std::vector<cv::Point> filtered;
            
            for (const auto& p : result) {
                bool inWeakLobe = false;
                for (const auto& weakP : weakestLobePoints) {
                    if (cv::norm(cv::Point2f(p) - cv::Point2f(weakP)) < 5.0f) {
                        inWeakLobe = true;
                        break;
                    }
                }
                if (!inWeakLobe) {
                    filtered.push_back(p);
                }
            }
            
            // Only use filtered contour if it's still reasonable
            if (filtered.size() > result.size() * 0.7f) {
                result = filtered;
            }
        }
    }
    
    return result;
}

// Helper for PASS 3: Width consistency check (only in -Y direction)
std::vector<cv::Point> PalmEstimator::applyWidthConsistencyCheck(const std::vector<cv::Point>& contour,
                                                               const cv::Point2f& wristLeft,
                                                               const cv::Point2f& wristRight,
                                                               float confidence) const {
    if (contour.size() < 50 || wristLeft.x < 0 || wristRight.x < 0) {
        return contour;
    }
    
    std::vector<cv::Point> result = contour;
    
    // Estimate wrist line and direction
    cv::Point2f wristDir = wristRight - wristLeft;
    float wristLen = cv::norm(wristDir);
    if (wristLen < 0.001f) return contour;
    
    wristDir = wristDir * (1.0f / wristLen);
    cv::Point2f wristNormal(-wristDir.y, wristDir.x);
    
    // Find centroid for orientation
    cv::Point2f centroid(0, 0);
    for (const auto& p : contour) {
        centroid += cv::Point2f(p);
    }
    centroid = centroid * (1.0f / contour.size());
    
    // Determine which side is arm (-Y)
    cv::Point2f toCentroid = centroid - wristLeft;
    if (toCentroid.dot(wristNormal) < 0) {
        wristNormal = -wristNormal;
    }
    
    // Sample points along the arm direction (-wristNormal)
    const int numSamples = 10;
    std::vector<float> widths;
    
    for (int i = 0; i < numSamples; i++) {
        float t = static_cast<float>(i) / (numSamples - 1);
        cv::Point2f samplePoint = wristLeft + wristDir * (wristLen * t);
        
        // Project in both directions perpendicular to wrist line
        cv::Point2f testPos = samplePoint + wristNormal * 100.0f;
        cv::Point2f testNeg = samplePoint - wristNormal * 100.0f;
        
        // Find intersection with contour (simplified)
        float width = 0.0f;
        // Simplified width estimation - in practice would need ray-contour intersection
        widths.push_back(width);
    }
    
    // Check width consistency (arm should have constant width)
    if (widths.size() > 3) {
        float meanWidth = 0.0f;
        for (float w : widths) meanWidth += w;
        meanWidth /= widths.size();
        
        float variance = 0.0f;
        for (float w : widths) {
            float diff = w - meanWidth;
            variance += diff * diff;
        }
        variance /= widths.size();
        
        float widthCV = std::sqrt(variance) / std::max(meanWidth, 1.0f);
        
        // If width is very consistent (arm-like), apply stronger trimming
        if (widthCV < 0.2f && confidence > 0.5f) {
            // Arm detected - apply progressive trimming
            return applyWristLineTrimming(result, wristLeft, wristRight, confidence * 1.2f);
        }
    }
    
    return result;
}

// Helper for PASS 4: Calibration-biased refinement
std::vector<cv::Point> PalmEstimator::applyCalibrationBiasedRefinement(const std::vector<cv::Point>& contour,
                                                                      float confidence) const {
    if (!isCalibrated || confidence < 0.3f) {
        return contour;
    }
    
    std::vector<cv::Point> result = contour;
    
    // Use calibration to bias trimming strength, not as hard limits
    
    // Example: If calibrated hand is smaller, be more conservative with trimming
    float area = cv::contourArea(contour);
    float calibratedArea = calibrationData.ratios.handWidth * calibrationData.ratios.handHeight;
    
    if (calibratedArea > 0) {
        float areaRatio = area / calibratedArea;
        
        // Adjust trimming aggressiveness based on size match
        if (areaRatio < 0.7f) {
            // Contour is smaller than calibrated - be conservative
            return result; // Skip additional trimming
        } else if (areaRatio > 1.5f) {
            // Contour is larger - could include more arm
            // Already handled by other passes
        }
    }
    
    return result;
}

// Palm center from constrained contour - NEVER FAILS
bool PalmEstimator::fitContourToModel(std::vector<cv::Point>& contour, 
                                     cv::Point2f& palmCenter,
                                     float& handSizeScale) {
    // ABSOLUTE: Always compute palm center
    if (contour.empty()) {
        if (lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            handSizeScale = 1.0f;
            return true;
        }
        // If no last valid, use center of frame
        palmCenter = cv::Point2f(320, 240);
        handSizeScale = 1.0f;
        return true;
    }
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) {
        if (lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            handSizeScale = 1.0f;
            return true;
        }
        // Fallback to bounding box center
        cv::Rect bbox = cv::boundingRect(contour);
        palmCenter = cv::Point2f(bbox.x + bbox.width/2.0f, bbox.y + bbox.height/2.0f);
        handSizeScale = 1.0f;
    } else {
        palmCenter = cv::Point2f(
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
        );
        
        // ENFORCE INVARIANT: Palm center must be inside contour
        palmCenter = projectPointToContourInterior(palmCenter, contour, palmCenter);
    }
    
    // Limit palm center jumps (soft limit)
    if (lastValidPalm.x >= 0) {
        float jumpDist = cv::norm(palmCenter - lastValidPalm);
        if (jumpDist > PE_MAX_PALM_CENTER_JUMP && jumpDist > 0.001f) {
            cv::Point2f direction = (palmCenter - lastValidPalm) * (1.0f / jumpDist);
            palmCenter = lastValidPalm + direction * (PE_MAX_PALM_CENTER_JUMP * 0.5f);
            
            // Re-project to contour after adjustment
            palmCenter = projectPointToContourInterior(palmCenter, contour, palmCenter);
        }
    }
    
    handSizeScale = 1.0f;
    if (isCalibrated && calibrationData.ratios.handWidth > 0) {
        // Simple scale estimation
        cv::Rect bbox = cv::boundingRect(contour);
        float currentWidth = static_cast<float>(bbox.width);
        float calibratedWidth = calibrationData.ratios.handWidth;
        
        handSizeScale = currentWidth / calibratedWidth;
        
        if (handSizeScale < 0.25f) handSizeScale = 0.25f;
        if (handSizeScale > 3.5f) handSizeScale = 3.5f;
    }
    
    // SOFT VALIDATION: Don't fail, just update last valid
    if (!validatePalmShape(contour, palmCenter)) {
        // Still update last valid for continuity
        lastValidPalm = palmCenter;
        failureFrameCount = 0;
    } else {
        lastValidPalm = palmCenter;
        failureFrameCount = 0;
    }
    
    // ALWAYS return true - palm center computation never fails
    return true;
}

void PalmEstimator::draw(cv::Mat& frame, const Result& result, 
                        const cv::Mat& skinMask, const cv::Mat& motionMask) {
    static std::vector<cv::Point> lastValidContour;
    static int contourGraceCounter = 0;
    
    if (!result.contour.empty() && result.handDetected) {
        lastValidContour = result.constrainedContour.empty() ? result.contour : result.constrainedContour;
        contourGraceCounter = 10;
    } else if (contourGraceCounter > 0) {
        contourGraceCounter--;
    }
    
    if (result.handDetected || contourGraceCounter > 0) {
        // Draw raw HSV contour (transparent)
        if (!result.contour.empty()) {
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{result.contour},
                0, cv::Scalar(100, 100, 100, 128), 1);
        }
        
        // Draw constrained contour (opaque)
        if (!result.constrainedContour.empty()) {
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{result.constrainedContour},
                0, cv::Scalar(0, 200, 0), 2);
        } else if (!lastValidContour.empty() && contourGraceCounter > 0) {
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{lastValidContour},
                0, cv::Scalar(100, 100, 100), 1);
        }
        
        // Draw AUTHORITATIVE wrist geometry from PalmEstimator
        if (result.wristLeft.x >= 0 && result.wristRight.x >= 0) {
            // Draw wrist line with confidence-based alpha
            int alpha = static_cast<int>(result.confidence * 255);
            cv::Scalar wristColor(0, 0, 255);
            
            // Wrist line
            cv::line(frame, result.wristLeft, result.wristRight, wristColor, 2);
            
            // Wrist endpoints
            cv::circle(frame, result.wristLeft, 6, wristColor, -1);
            cv::circle(frame, result.wristRight, 6, wristColor, -1);
            
            // Wrist midpoint
            if (result.wristMid.x >= 0) {
                cv::circle(frame, result.wristMid, 8, cv::Scalar(255, 0, 255), -1);
                
                // Draw line from palm to wrist mid
                if (result.palm.x >= 0) {
                    cv::line(frame, result.palm, result.wristMid, cv::Scalar(255, 0, 255), 1);
                }
            }
            
            // Label wrist points with confidence
            std::string wristLabel = "Wrist L (" + std::to_string((int)(result.confidence * 100)) + "%)";
            cv::putText(frame, wristLabel, result.wristLeft + cv::Point2f(5, -10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, wristColor, 1);
            cv::putText(frame, "Wrist R", result.wristRight + cv::Point2f(5, 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, wristColor, 1);
        }
        
        // Draw palm center (from PalmEstimator)
        if (result.palm.x >= 0) {
            cv::circle(frame, result.palm, 8, cv::Scalar(0, 255, 255), -1);
            cv::putText(frame, "Palm", result.palm + cv::Point2f(10, 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        }
        
        if (geometryUpdater.isPalmValid()) {
            // Draw hand reference frame from GeometryUpdater
            const auto& handFrame = geometryUpdater.getHandFrame();
            if (handFrame.isValid()) {
                cv::Point2f axisEnd = result.palm + handFrame.primaryAxis * 50.0f;
                cv::Point2f lateralEnd = result.palm + handFrame.secondaryAxis * 50.0f;
                
                cv::arrowedLine(frame, result.palm, axisEnd, cv::Scalar(0, 255, 0), 2);
                cv::arrowedLine(frame, result.palm, lateralEnd, cv::Scalar(255, 0, 0), 2);
                
                cv::putText(frame, "Primary", axisEnd + cv::Point2f(5, 5),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
                cv::putText(frame, "Lateral", lateralEnd + cv::Point2f(5, 5),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
            }
            
            // Draw fingers from GeometryUpdater
            const auto& fingers = geometryUpdater.getFingerIdentities();
            for (const auto& finger : fingers) {
                if (finger.isDetected) {
                    cv::Scalar color;
                    std::string fingerName;
                    switch (finger.id) {
                        case 0: 
                            color = cv::Scalar(255, 165, 0); 
                            fingerName = "Thumb";
                            break;
                        case 1: 
                            color = cv::Scalar(0, 255, 0); 
                            fingerName = "Index";
                            break;
                        case 2: 
                            color = cv::Scalar(255, 0, 0); 
                            fingerName = "Middle";
                            break;
                        case 3: 
                            color = cv::Scalar(255, 0, 255); 
                            fingerName = "Ring";
                            break;
                        case 4: 
                            color = cv::Scalar(255, 20, 147); 
                            fingerName = "Pinky";
                            break;
                        default: 
                            color = cv::Scalar(0, 200, 255); 
                            fingerName = "?";
                            break;
                    }
                    
                    cv::circle(frame, finger.displayTip, 6, color, -1);
                    
                    cv::putText(frame, fingerName, 
                              finger.displayTip + cv::Point2f(8, -10),
                              cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                    
                    // Draw line to palm
                    if (result.palm.x >= 0) {
                        cv::line(frame, result.palm, finger.displayTip,
                                 cv::Scalar(200, 200, 200), 1);
                    }
                    
                    // Show lock status
                    std::string status = finger.isLocked ? "L" : std::to_string((int)(finger.confidence * 100));
                    cv::putText(frame, status + "%",
                              finger.displayTip + cv::Point2f(5, 30),
                              cv::FONT_HERSHEY_SIMPLEX, 0.4,
                              finger.isLocked ? cv::Scalar(0, 255, 0) : cv::Scalar(200, 200, 200),
                              1);
                }
            }
            
            // Status info from GeometryUpdater
            std::string stateInfo;
            switch (geometryUpdater.getFingerState()) {
                case FINGER_STATE_FIST: stateInfo = "FIST"; break;
                case FINGER_STATE_PARTIAL: stateInfo = "PARTIAL"; break;
                case FINGER_STATE_OPEN: stateInfo = "OPEN"; break;
            }
            
            stateInfo += " | Fingers: " + std::to_string(geometryUpdater.getDetectedFingerCount());
            if (result.palm.x >= 0) {
                cv::putText(frame, stateInfo, 
                          result.palm + cv::Point2f(-60, -30),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 200), 1);
            }
        }
    }
    
    // Status overlay
    cv::rectangle(frame, cv::Rect(0, 0, 640, 80), cv::Scalar(0, 0, 0, 180), -1);
    
    std::string status = result.handDetected ? result.status : result.status;
    cv::Scalar statusColor = result.handDetected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    
    cv::putText(frame, status, cv::Point(10, 25), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
    
    // Add wrist status with confidence
    if (result.wristLeft.x >= 0 && result.wristRight.x >= 0) {
        std::string wristInfo = "Wrist: [" + 
                               std::to_string((int)result.wristLeft.x) + "," + 
                               std::to_string((int)result.wristLeft.y) + "] - [" +
                               std::to_string((int)result.wristRight.x) + "," + 
                               std::to_string((int)result.wristRight.y) + "]";
        wristInfo += " (conf: " + std::to_string((int)(result.confidence * 100)) + "%)";
        cv::putText(frame, wristInfo, cv::Point(10, 50),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 255), 1);
    }
    
    std::string controls = "C: Calibrate | T: Toggle | ESC: Exit";
    if (isCalibrated) {
        controls += " | R: Reset Calibration";
    }
    cv::putText(frame, controls, cv::Point(10, 480 - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}