#include "PalmEstimator.h"
#include <algorithm>
#include <iostream>

// External declarations from main.cpp
extern cv::Point2f lastValidPalm;
extern cv::Point2f lastValidThumbBase;
extern cv::Point2f lastValidPinkyBase;
extern int failureFrameCount;
extern std::atomic<bool> isCalibrated;  // Changed from bool to std::atomic<bool>
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
        
        if (!fitContourToModel(result.contour, result.palm, result.thumbBase, 
                              result.pinkyBase, result.handSizeScale)) {
            result.status = "Geometry fitting failed, using HSV";
        }
        
        result.fingerTips = detectFingerTips(result.contour, result.palm);
        
        geometryUpdater.updateGeometry(result.palm, result.thumbBase, result.pinkyBase,
                                     result.fingerTips, result.handSizeScale, 
                                     result.contour);
        
        if (geometryUpdater.isPalmValid()) {
            result.smoothedPalm = geometryUpdater.getPalmCenter();
            result.handDetected = true;
        }
        
        float anatConfidence = geometryUpdater.getAnatomicalConfidence();
        float overallConfidence = geometryUpdater.getOverallConfidence();
        
        std::string stateStr;
        switch (geometryUpdater.getFingerState()) {
            case FINGER_STATE_FIST: stateStr = " (fist)"; break;
            case FINGER_STATE_PARTIAL: stateStr = " (partial)"; break;
            case FINGER_STATE_OPEN: stateStr = " (open)"; break;
        }
        
        if (anatConfidence > 0.7f) {
            result.status = "Hand (stable)" + stateStr;
            result.confidence = overallConfidence;
        } else if (anatConfidence > 0.3f) {
            result.status = "Hand (correcting)" + stateStr;
            result.confidence = overallConfidence * 0.7f;
        } else {
            result.status = "Hand (low anatomy)" + stateStr;
            result.confidence = overallConfidence * 0.4f;
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
                                     cv::Point2f& thumbBase,
                                     cv::Point2f& pinkyBase,
                                     float& handSizeScale) {
    if (contour.size() < 8) {
        if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            thumbBase = lastValidThumbBase;
            pinkyBase = lastValidPinkyBase;
            handSizeScale = 1.0f;
            failureFrameCount++;
            return true;
        }
    }
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) {
        if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            thumbBase = lastValidThumbBase;
            pinkyBase = lastValidPinkyBase;
            handSizeScale = 1.0f;
            failureFrameCount++;
            return true;
        }
        if (!contour.empty()) {
            cv::Rect bbox = cv::boundingRect(contour);
            palmCenter = cv::Point2f(bbox.x + bbox.width/2.0f, bbox.y + bbox.height/2.0f);
        }
    } else {
        palmCenter = cv::Point2f(
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
        );
    }
    
    if (lastValidPalm.x >= 0) {
        float jumpDist = cv::norm(palmCenter - lastValidPalm);
        if (jumpDist > PE_MAX_PALM_CENTER_JUMP && jumpDist > 0.001f) {
            cv::Point2f direction = (palmCenter - lastValidPalm) * (1.0f / jumpDist);
            palmCenter = lastValidPalm + direction * (PE_MAX_PALM_CENTER_JUMP * 0.5f);
        }
    }
    
    if (contour.size() > 8) {
        cv::Point leftmost = contour[0];
        cv::Point rightmost = contour[0];
        
        for (const auto& p : contour) {
            if (p.x < leftmost.x) leftmost = p;
            if (p.x > rightmost.x) rightmost = p;
        }
        
        thumbBase = cv::Point2f(static_cast<float>(leftmost.x), static_cast<float>(leftmost.y));
        pinkyBase = cv::Point2f(static_cast<float>(rightmost.x), static_cast<float>(rightmost.y));
    }
    
    handSizeScale = 1.0f;
    if (isCalibrated && calibrationData.ratios.handWidth > 0) {
        float currentWidth = cv::norm(thumbBase - pinkyBase);
        float calibratedWidth = calibrationData.ratios.handWidth;
        
        float lowerBound = calibratedWidth * PE_DISTANCE_LOWER_BOUND;
        float upperBound = calibratedWidth * PE_DISTANCE_UPPER_BOUND;
        
        float clampedWidth = currentWidth;
        if (clampedWidth < lowerBound) {
            clampedWidth = lowerBound * 0.3f + currentWidth * 0.7f;
        }
        if (clampedWidth > upperBound) {
            clampedWidth = upperBound * 0.3f + currentWidth * 0.7f;
        }
        
        handSizeScale = clampedWidth / calibratedWidth;
        
        if (handSizeScale < 0.25f) handSizeScale = 0.25f;
        if (handSizeScale > 3.5f) handSizeScale = 3.5f;
    }
    
    if (!validatePalmShape(contour, palmCenter)) {
        if (failureFrameCount < PE_MAX_FAILURE_FRAMES && lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            thumbBase = lastValidThumbBase;
            pinkyBase = lastValidPinkyBase;
            handSizeScale = 1.0f;
            failureFrameCount++;
            return true;
        }
        return false;
    }
    
    lastValidPalm = palmCenter;
    lastValidThumbBase = thumbBase;
    lastValidPinkyBase = pinkyBase;
    failureFrameCount = 0;
    
    return true;
}

std::vector<cv::Point2f> PalmEstimator::detectFingerTips(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter) {
    std::vector<cv::Point2f> fingerTips;
    
    if (contour.empty() || palmCenter.x < 0) return fingerTips;
    
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull, false);
    
    std::vector<cv::Point2f> candidates;
    for (const auto& p : hull) {
        cv::Point2f point(p);
        float dist = cv::norm(point - palmCenter);
        
        float minDist = (isCalibrated ? calibrationData.ratios.palmRadius * 0.8f : 30.0f);
        float maxDist = (isCalibrated ? calibrationData.ratios.maxFingerLength * 1.5f : 150.0f);
        
        if (dist >= minDist && dist <= maxDist) {
            if (point.y < palmCenter.y + dist * 0.5f) {
                candidates.push_back(point);
            }
        }
    }
    
    std::sort(candidates.begin(), candidates.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
        });
    
    int maxFingers = std::min(5, static_cast<int>(candidates.size()));
    for (int i = 0; i < maxFingers; i++) {
        fingerTips.push_back(candidates[i]);
    }
    
    return fingerTips;
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
            
            const HandGeometryState& state = geometryUpdater.getState();
            
            cv::circle(frame, state.smoothedPalmCenter, 8, cv::Scalar(0, 255, 255), -1);
            
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
                    
                    float dist = cv::norm(finger.displayTip - state.smoothedPalmCenter);
                    cv::line(frame, state.smoothedPalmCenter, finger.displayTip,
                             cv::Scalar(200, 200, 200), 1);
                    
                    cv::putText(frame,
                        std::to_string((int)dist),
                        finger.displayTip + cv::Point2f(5, 15),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.4,
                        cv::Scalar(255, 255, 255),
                        1
                    );
                    
                    std::string status = finger.isLocked ? "L" : std::to_string((int)(finger.confidence * 100));
                    cv::putText(frame,
                        status + "%",
                        finger.displayTip + cv::Point2f(5, 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.4,
                        finger.isLocked ? cv::Scalar(0, 255, 0) : cv::Scalar(200, 200, 200),
                        1
                    );
                }
            }
            
            if (state.smoothedWristLeft.x >= 0 && state.smoothedWristRight.x >= 0) {
                cv::line(frame, state.smoothedWristLeft, state.smoothedWristRight, 
                        cv::Scalar(255, 0, 0), 2);
                
                cv::circle(frame, state.smoothedWristLeft, 8, cv::Scalar(255, 0, 0), -1);
                cv::circle(frame, state.smoothedWristRight, 8, cv::Scalar(255, 0, 0), -1);
                cv::circle(frame, state.wristMid, 6, cv::Scalar(0, 255, 255), -1);
                
                cv::putText(frame, "Wrist L", 
                          state.smoothedWristLeft + cv::Point2f(10, 0),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
                cv::putText(frame, "Wrist R", 
                          state.smoothedWristRight + cv::Point2f(10, 0),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
                cv::putText(frame, "Wrist Mid", 
                          state.wristMid + cv::Point2f(10, 0),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                
                if (cv::norm(state.lastWristDir) > 0.001f) {
                    cv::Point2f wristDir = state.lastWristDir;
                    cv::Point2f lateralDir(-wristDir.y, wristDir.x);
                    cv::Point2f cutoffStart = state.wristMid - lateralDir * 100;
                    cv::Point2f cutoffEnd = state.wristMid + lateralDir * 100;
                    cv::line(frame, cutoffStart, cutoffEnd, 
                            cv::Scalar(255, 100, 100), 1, cv::LINE_AA);
                }
            }
            
            std::string stateInfo;
            switch (geometryUpdater.getFingerState()) {
                case FINGER_STATE_FIST: stateInfo = "FIST"; break;
                case FINGER_STATE_PARTIAL: stateInfo = "PARTIAL"; break;
                case FINGER_STATE_OPEN: stateInfo = "OPEN"; break;
            }
            stateInfo += " | Fingers: " + std::to_string(geometryUpdater.getDetectedFingerCount());
            stateInfo += " | Conf: " + std::to_string((int)(geometryUpdater.getOverallConfidence() * 100)) + "%";
            stateInfo += " | Sm: " + std::to_string((int)(state.currentSmoothingAlpha * 100)) + "%";
            if (state.hsvIsRelaxed) {
                stateInfo += " HSV-R";
            }
            cv::putText(frame, stateInfo, 
                      state.smoothedPalmCenter + cv::Point2f(-60, -30),
                      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 200), 1);
        }
    }
    
    cv::rectangle(frame, cv::Rect(0, 0, 640, 80), cv::Scalar(0, 0, 0, 180), -1);
    
    std::string status = result.handDetected ? result.status : result.status;
    cv::Scalar statusColor = result.handDetected ? 
        (geometryUpdater.getAnatomicalConfidence() > 0.7f ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 165, 0)) : 
        cv::Scalar(0, 0, 255);
    
    cv::putText(frame, status, cv::Point(10, 25), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
    
    if (result.handDetected) {
        std::string info = "Area: " + std::to_string((int)result.area) +
                          " | Fingers: " + std::to_string(geometryUpdater.getFingerIdentities().size()) +
                          " | Anat: " + std::to_string((int)(geometryUpdater.getAnatomicalConfidence() * 100)) + "%";
        cv::putText(frame, info, cv::Point(10, 50),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 200), 1);
    }
    
    std::string cursorStatus = true ? "CURSOR: ON" : "CURSOR: OFF";
    cv::Scalar cursorColor = true ? cv::Scalar(0, 255, 0) : cv::Scalar(100, 100, 255);
    cv::putText(frame, cursorStatus, cv::Point(640 - 120, 50),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cursorColor, 1);
    
    std::string controls = "C: Calibrate | T: Toggle | ESC: Exit";
    if (isCalibrated) {
        controls += " | R: Reset Calibration";
    }
    cv::putText(frame, controls, cv::Point(10, 480 - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}