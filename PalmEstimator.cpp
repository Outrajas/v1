#include "PalmEstimator.h"
#include <algorithm>
#include <iostream>

// External declarations from main.cpp
extern cv::Point2f lastValidPalm;
extern cv::Point2f lastValidThumbBase;
extern cv::Point2f lastValidPinkyBase;
extern int failureFrameCount;
extern std::atomic<bool> isCalibrated;
extern CalibrationResult calibrationData;
extern GeometryUpdater geometryUpdater;
extern ShapeAnchoredTracker shapeTracker;

PalmEstimator::Result PalmEstimator::detect(const cv::Mat& frame, const cv::Mat& motionMask, 
                                           const cv::Mat& skinMask, float hsvConfidence) {
    Result result;
    result.status = "Processing";
    
    try {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        result.contoursFound = contours.size();
        
        if (contours.empty()) {
            if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
                result.handDetected = true;
                result.palm = lastValidPalm;
                result.status = "Hand (fallback)";
                failureFrameCount++;
                return result;
            }
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
            if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
                result.handDetected = true;
                result.palm = lastValidPalm;
                result.status = "Hand (fallback)";
                failureFrameCount++;
                return result;
            }
            result.status = "No potential hands";
            return result;
        }
        
        auto maxScoreIt = std::max_element(potentialScores.begin(), potentialScores.end());
        size_t bestIdxInPotentials = std::distance(potentialScores.begin(), maxScoreIt);
        size_t bestContourIdx = potentialIndices[bestIdxInPotentials];
        
        result.contour = contours[bestContourIdx];
        result.area = cv::contourArea(result.contour);
        result.boundingBox = cv::boundingRect(result.contour);
        
        float aspect, solidity;
        couldBeHand(result.contour, aspect, solidity);
        result.aspectRatio = aspect;
        result.solidity = solidity;
        
        // Fit contour and estimate wrist position
        cv::Point2f wristMid;
        if (!fitContourToModel(result.contour, result.palm, wristMid, result.handSizeScale)) {
            result.status = "Geometry fitting failed";
        } else {
            // Store wrist position
            result.wristMid = wristMid;
        }
        
        // Update geometry with independent paths
        geometryUpdater.updateGeometry(result.palm, wristMid, 
                                      cv::Point2f(-1, -1), cv::Point2f(-1, -1),
                                      result.contour);
        
        if (geometryUpdater.isPalmValid()) {
            result.smoothedPalm = geometryUpdater.getPalmCenter();
            result.smoothedWristMid = geometryUpdater.getWristMid();
            result.handDetected = true;
            
            // Get finger state
            result.fingerState = geometryUpdater.getFingerState();
            result.detectedFingerCount = geometryUpdater.getDetectedFingerCount();
            
            // Build status string
            std::string stateStr;
            switch (result.fingerState) {
                case FINGER_STATE_FIST: stateStr = " (fist)"; break;
                case FINGER_STATE_PARTIAL: stateStr = " (partial)"; break;
                case FINGER_STATE_OPEN: stateStr = " (open)"; break;
            }
            
            result.status = "Hand" + stateStr;
            result.confidence = 0.8f;  // Basic confidence
        }
        
    } catch (const cv::Exception& e) {
        result.status = "Error";
        if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
            result.handDetected = true;
            result.palm = lastValidPalm;
            result.status = "Hand (recovery)";
            failureFrameCount++;
        }
    }
    
    return result;
}

bool PalmEstimator::validatePalmShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter) {
    if (contour.size() < 20) return true;
    
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
        if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
            return true;
        }
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
        
        float geometryScore = computeGeometryConsistencyScore(contour, contourCenter, 
            std::sqrt(area / M_PI));
        
        return geometryScore > 0.2f;
    }
    
    return true;
}

bool PalmEstimator::fitContourToModel(std::vector<cv::Point>& contour, 
                                     cv::Point2f& palmCenter,
                                     cv::Point2f& wristMid,
                                     float& handSizeScale) {
    if (contour.size() < 8) {
        if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            wristMid = cv::Point2f(palmCenter.x, palmCenter.y + 100);  // Default below palm
            handSizeScale = 1.0f;
            failureFrameCount++;
            return true;
        }
    }
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) {
        if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            wristMid = cv::Point2f(palmCenter.x, palmCenter.y + 100);
            handSizeScale = 1.0f;
            failureFrameCount++;
            return true;
        }
        if (!contour.empty()) {
            cv::Rect bbox = cv::boundingRect(contour);
            palmCenter = cv::Point2f(bbox.x + bbox.width/2.0f, bbox.y + bbox.height/2.0f);
            wristMid = cv::Point2f(palmCenter.x, bbox.y + bbox.height);
        }
    } else {
        palmCenter = cv::Point2f(
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
        );
        
        // Estimate wrist position as lowest point of contour
        int lowestY = contour[0].y;
        cv::Point lowestPoint = contour[0];
        
        for (const auto& p : contour) {
            if (p.y > lowestY) {
                lowestY = p.y;
                lowestPoint = p;
            }
        }
        
        wristMid = cv::Point2f(static_cast<float>(lowestPoint.x), 
                              static_cast<float>(lowestPoint.y));
        
        // Adjust wrist to be below palm
        if (wristMid.y < palmCenter.y + 20) {
            wristMid.y = palmCenter.y + 100;  // Force reasonable distance
        }
    }
    
    // Limit palm center jumps
    if (lastValidPalm.x >= 0) {
        float jumpDist = cv::norm(palmCenter - lastValidPalm);
        if (jumpDist > PE_MAX_PALM_CENTER_JUMP && jumpDist > 0.001f) {
            cv::Point2f direction = (palmCenter - lastValidPalm) * (1.0f / jumpDist);
            palmCenter = lastValidPalm + direction * (PE_MAX_PALM_CENTER_JUMP * 0.5f);
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
    
    if (!validatePalmShape(contour, palmCenter)) {
        if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            wristMid = cv::Point2f(palmCenter.x, palmCenter.y + 100);
            handSizeScale = 1.0f;
            failureFrameCount++;
            return true;
        }
        return false;
    }
    
    lastValidPalm = palmCenter;
    failureFrameCount = 0;
    
    return true;
}

void PalmEstimator::draw(cv::Mat& frame, const Result& result, 
                        const cv::Mat& skinMask, const cv::Mat& motionMask) {
    static std::vector<cv::Point> lastValidContour;
    static int contourGraceCounter = 0;
    
    if (!result.contour.empty() && result.handDetected) {
        lastValidContour = result.contour;
        contourGraceCounter = 10;
    } else if (contourGraceCounter > 0) {
        contourGraceCounter--;
    }
    
    if (result.handDetected || contourGraceCounter > 0) {
        if (geometryUpdater.isPalmValid()) {
            const auto& palmContour = geometryUpdater.getPalmContour();
            if (!palmContour.empty()) {
                cv::drawContours(frame, std::vector<std::vector<cv::Point>>{palmContour},
                    0, cv::Scalar(0, 200, 0), 2);
            } else if (!lastValidContour.empty() && contourGraceCounter > 0) {
                cv::drawContours(frame, std::vector<std::vector<cv::Point>>{lastValidContour},
                    0, cv::Scalar(100, 100, 100), 1);
            }
            
            // Draw palm center
            cv::Point2f palmCenter = geometryUpdater.getPalmCenter();
            cv::circle(frame, palmCenter, 8, cv::Scalar(0, 255, 255), -1);
            
            // Draw wrist
            cv::Point2f wristMid = geometryUpdater.getWristMid();
            if (wristMid.x >= 0) {
                cv::circle(frame, wristMid, 8, cv::Scalar(255, 0, 0), -1);
                cv::line(frame, palmCenter, wristMid, cv::Scalar(255, 0, 0), 2);
                
                // Draw hand reference frame
                const auto& handFrame = geometryUpdater.getHandFrame();
                cv::Point2f axisEnd = palmCenter + handFrame.primaryAxis * 50.0f;
                cv::Point2f lateralEnd = palmCenter + handFrame.secondaryAxis * 50.0f;
                
                cv::arrowedLine(frame, palmCenter, axisEnd, cv::Scalar(0, 255, 0), 2);
                cv::arrowedLine(frame, palmCenter, lateralEnd, cv::Scalar(255, 0, 0), 2);
                
                cv::putText(frame, "Primary", axisEnd + cv::Point2f(5, 5),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
                cv::putText(frame, "Lateral", lateralEnd + cv::Point2f(5, 5),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
            }
            
            // Draw fingers
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
                    cv::line(frame, palmCenter, finger.displayTip,
                             cv::Scalar(200, 200, 200), 1);
                    
                    // Show lock status
                    std::string status = finger.isLocked ? "L" : std::to_string((int)(finger.confidence * 100));
                    cv::putText(frame, status + "%",
                              finger.displayTip + cv::Point2f(5, 30),
                              cv::FONT_HERSHEY_SIMPLEX, 0.4,
                              finger.isLocked ? cv::Scalar(0, 255, 0) : cv::Scalar(200, 200, 200),
                              1);
                }
            }
            
            // Draw wrist boundaries
            cv::Point2f wristLeft = geometryUpdater.getWristLeft();
            cv::Point2f wristRight = geometryUpdater.getWristRight();
            
            if (wristLeft.x >= 0 && wristRight.x >= 0) {
                cv::line(frame, wristLeft, wristRight, cv::Scalar(255, 0, 0), 2);
                cv::circle(frame, wristLeft, 6, cv::Scalar(255, 0, 0), -1);
                cv::circle(frame, wristRight, 6, cv::Scalar(255, 0, 0), -1);
            }
            
            // Status info
            std::string stateInfo;
            switch (geometryUpdater.getFingerState()) {
                case FINGER_STATE_FIST: stateInfo = "FIST"; break;
                case FINGER_STATE_PARTIAL: stateInfo = "PARTIAL"; break;
                case FINGER_STATE_OPEN: stateInfo = "OPEN"; break;
            }
            
            stateInfo += " | Fingers: " + std::to_string(geometryUpdater.getDetectedFingerCount());
            cv::putText(frame, stateInfo, 
                      palmCenter + cv::Point2f(-60, -30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 200), 1);
        }
    }
    
    // Status overlay
    cv::rectangle(frame, cv::Rect(0, 0, 640, 80), cv::Scalar(0, 0, 0, 180), -1);
    
    std::string status = result.handDetected ? result.status : result.status;
    cv::Scalar statusColor = result.handDetected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    
    cv::putText(frame, status, cv::Point(10, 25), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
    
    if (result.handDetected) {
        std::string info = "Area: " + std::to_string((int)result.area) +
                          " | Fingers: " + std::to_string(geometryUpdater.getDetectedFingerCount());
        cv::putText(frame, info, cv::Point(10, 50),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 200), 1);
    }
    
    std::string controls = "C: Calibrate | T: Toggle | ESC: Exit";
    if (isCalibrated) {
        controls += " | R: Reset Calibration";
    }
    cv::putText(frame, controls, cv::Point(10, 480 - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}