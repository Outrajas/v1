#include "ManualCalibrator.h"
#include <iostream>

extern CalibrationResult calibrationData;
extern std::atomic<bool> isCalibrated;  // Changed from bool to std::atomic<bool>
extern GeometryUpdater geometryUpdater;
extern ShapeAnchoredTracker shapeTracker;

void EnhancedManualCalibrator::startCalibration(const cv::Mat& frame) {
    frame.copyTo(calibrationFrame);
    cv::cvtColor(calibrationFrame, calibrationFrameHSV, cv::COLOR_BGR2HSV);
    
    hsvRectStart = cv::Point(-1, -1);
    hsvSamplingRect = cv::Rect();
    palmTrace.clear();
    thumbBase = cv::Point2f(-1, -1);
    pinkyBase = cv::Point2f(-1, -1);
    wristLeft = cv::Point2f(-1, -1);
    wristRight = cv::Point2f(-1, -1);
    
    for (int i = 0; i < 5; i++) {
        fingerTips[i] = cv::Point2f(-1, -1);
        fingerBases[i] = cv::Point2f(-1, -1);
        fingerTipsMarked[i] = false;
        fingerBasesMarked[i] = false;
    }
    wristLeftMarked = false;
    wristRightMarked = false;
    
    calibrationStep = 0;
    isActive = true;
    isDrawing = false;
    
    calibrationData = CalibrationResult();
    
    std::cout << "\n=== CALIBRATION ===" << std::endl;
    std::cout << "Step 1: Draw rectangle on palm for HSV sampling" << std::endl;
}

void EnhancedManualCalibrator::handleMouse(int event, int x, int y) {
    if (!isActive) return;
    
    switch (calibrationStep) {
        case 0:
            if (event == cv::EVENT_LBUTTONDOWN) {
                hsvRectStart = cv::Point(x, y);
                hsvSamplingRect = cv::Rect(x, y, 0, 0);
            }
            else if (event == cv::EVENT_MOUSEMOVE && hsvRectStart.x >= 0) {
                int width = std::abs(x - hsvRectStart.x);
                int height = std::abs(y - hsvRectStart.y);
                
                hsvSamplingRect = cv::Rect(
                    (hsvRectStart.x < x) ? hsvRectStart.x : x,
                    (hsvRectStart.y < y) ? hsvRectStart.y : y,
                    width,
                    height
                );
            }
            else if (event == cv::EVENT_LBUTTONUP) {
                hsvRectStart = cv::Point(-1, -1);
            }
            break;
            
        case 1:
            if (event == cv::EVENT_LBUTTONDOWN) {
                isDrawing = true;
                palmTrace.clear();
                palmTrace.push_back(cv::Point(x, y));
            }
            else if (event == cv::EVENT_MOUSEMOVE && isDrawing) {
                palmTrace.push_back(cv::Point(x, y));
            }
            else if (event == cv::EVENT_LBUTTONUP) {
                isDrawing = false;
            }
            break;
            
        case 2:
            if (event == cv::EVENT_LBUTTONDOWN) {
                for (int i = 0; i < 5; i++) {
                    if (!fingerBasesMarked[i]) {
                        fingerBases[i] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
                        fingerBasesMarked[i] = true;
                        std::cout << fingerNames[i] << " base marked" << std::endl;
                        
                        if (i == 0) thumbBase = fingerBases[i];
                        if (i == 4) pinkyBase = fingerBases[i];
                        break;
                    }
                }
            }
            break;
            
        case 3:
            if (event == cv::EVENT_LBUTTONDOWN) {
                for (int i = 0; i < 5; i++) {
                    if (!fingerTipsMarked[i]) {
                        fingerTips[i] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
                        fingerTipsMarked[i] = true;
                        std::cout << fingerNames[i] << " tip marked" << std::endl;
                        break;
                    }
                }
            }
            break;
            
        case 4:
            if (event == cv::EVENT_LBUTTONDOWN) {
                if (!wristLeftMarked) {
                    wristLeft = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
                    wristLeftMarked = true;
                    std::cout << "Left wrist boundary marked" << std::endl;
                }
                else if (!wristRightMarked) {
                    wristRight = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
                    wristRightMarked = true;
                    std::cout << "Right wrist boundary marked" << std::endl;
                }
            }
            break;
    }
}

bool EnhancedManualCalibrator::validateStep(int step) const {
    switch (step) {
        case 0: return hsvSamplingRect.area() >= 100;
        case 1: return palmTrace.size() >= 20;
        case 2: 
            for (int i = 0; i < 5; i++) {
                if (!fingerBasesMarked[i]) return false;
            }
            return true;
        case 3: {
            if (!fingerTipsMarked[0] || !fingerTipsMarked[1]) return false;
            int tipsCount = 0;
            for (int i = 0; i < 5; i++) {
                if (fingerTipsMarked[i]) tipsCount++;
            }
            return tipsCount >= 3;
        }
        case 4: return wristLeftMarked && wristRightMarked;
        case 5: return true;
        default: return false;
    }
}

void EnhancedManualCalibrator::drawStepUI(cv::Mat& frame) {
    cv::rectangle(frame, cv::Rect(0, 0, frame.cols, 80), cv::Scalar(40, 40, 40), cv::FILLED);
    
    std::string stepText;
    std::string instructionText;
    
    switch (calibrationStep) {
        case 0:
            stepText = "STEP 1: Select palm area for HSV";
            instructionText = "Draw rectangle on palm - Press N when done";
            if (!hsvSamplingRect.empty()) {
                cv::rectangle(frame, hsvSamplingRect, cv::Scalar(0, 255, 0), 2);
            }
            break;
        case 1:
            stepText = "STEP 2: Trace palm outline";
            instructionText = "Click and drag around palm - Press N when done";
            if (palmTrace.size() > 1) {
                for (size_t i = 1; i < palmTrace.size(); i++) {
                    cv::line(frame, palmTrace[i-1], palmTrace[i], cv::Scalar(0, 255, 0), 2);
                }
            }
            break;
        case 2:
            stepText = "STEP 3: Mark ALL 5 finger bases";
            instructionText = "Click in order: Thumb, Index, Middle, Ring, Pinky - Press N when done";
            for (int i = 0; i < 5; i++) {
                if (fingerBasesMarked[i]) {
                    cv::Scalar color;
                    switch (i) {
                        case 0: color = cv::Scalar(255, 165, 0); break;
                        case 1: color = cv::Scalar(0, 255, 0); break;
                        case 2: color = cv::Scalar(255, 0, 0); break;
                        case 3: color = cv::Scalar(255, 0, 255); break;
                        case 4: color = cv::Scalar(255, 20, 147); break;
                    }
                    cv::circle(frame, fingerBases[i], 8, color, -1);
                    cv::putText(frame, fingerNames[i], 
                               cv::Point(static_cast<int>(fingerBases[i].x + 10), 
                                        static_cast<int>(fingerBases[i].y - 10)),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                }
            }
            break;
        case 3:
            stepText = "STEP 4: Mark finger tips";
            instructionText = "Click finger tips (at least thumb, index, and one more) - Press N when done";
            for (int i = 0; i < 5; i++) {
                if (fingerTipsMarked[i]) {
                    cv::Scalar color;
                    switch (i) {
                        case 0: color = cv::Scalar(255, 165, 0); break;
                        case 1: color = cv::Scalar(0, 255, 0); break;
                        case 2: color = cv::Scalar(255, 0, 0); break;
                        case 3: color = cv::Scalar(255, 0, 255); break;
                        case 4: color = cv::Scalar(255, 20, 147); break;
                    }
                    cv::circle(frame, fingerTips[i], 6, color, -1);
                    if (fingerBasesMarked[i]) {
                        cv::line(frame, fingerBases[i], fingerTips[i], color, 1);
                    }
                }
            }
            break;
        case 4:
            stepText = "STEP 5: Mark wrist BOUNDARIES";
            instructionText = "Click LEFT then RIGHT wrist boundary points - Press N when done";
            if (wristLeftMarked) {
                cv::circle(frame, wristLeft, 10, cv::Scalar(255, 0, 0), -1);
                cv::putText(frame, "Wrist L", 
                           cv::Point(static_cast<int>(wristLeft.x + 15), 
                                    static_cast<int>(wristLeft.y)),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
            if (wristRightMarked) {
                cv::circle(frame, wristRight, 10, cv::Scalar(0, 0, 255), -1);
                cv::putText(frame, "Wrist R", 
                           cv::Point(static_cast<int>(wristRight.x + 15), 
                                    static_cast<int>(wristRight.y)),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
            if (wristLeftMarked && wristRightMarked) {
                cv::line(frame, wristLeft, wristRight, cv::Scalar(255, 200, 0), 3);
                cv::Point2f wristMid = (wristLeft + wristRight) * 0.5f;
                cv::circle(frame, wristMid, 6, cv::Scalar(255, 255, 0), -1);
            }
            break;
        case 5:
            stepText = "STEP 6: Ready to finalize";
            instructionText = "Press F to finish calibration";
            break;
    }
    
    cv::putText(frame, stepText, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(frame, instructionText, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    
    std::string progress = "Step " + std::to_string(calibrationStep + 1) + "/6";
    cv::putText(frame, progress, cv::Point(frame.cols - 100, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::putText(frame, "N: Next Step | F: Finish | C: Clear | ESC: Cancel", 
               cv::Point(10, frame.rows - 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
}

bool EnhancedManualCalibrator::processCalibration(int key, cv::Mat& displayFrame) {
    if (!isActive) return false;
    
    calibrationFrame.copyTo(displayFrame);
    drawStepUI(displayFrame);
    
    if (key == 'n' || key == 'N') {
        if (validateStep(calibrationStep)) {
            calibrationStep++;
            if (calibrationStep >= 6) {
                return finalizeCalibration();
            } else {
                switch (calibrationStep) {
                    case 1: std::cout << "Step 2: Trace palm outline" << std::endl; break;
                    case 2: std::cout << "Step 3: Mark finger bases" << std::endl; break;
                    case 3: std::cout << "Step 4: Mark finger tips" << std::endl; break;
                    case 4: std::cout << "Step 5: Mark wrist BOUNDARIES" << std::endl; break;
                    case 5: std::cout << "Step 6: Ready to finalize" << std::endl; break;
                }
            }
        } else {
            std::cout << "Please complete current step first" << std::endl;
        }
    } 
    else if (key == 'f' || key == 'F') {
        return finalizeCalibration();
    }
    else if (key == 'c' || key == 'C') {
        switch (calibrationStep) {
            case 0: 
                hsvSamplingRect = cv::Rect(); 
                std::cout << "HSV rectangle cleared" << std::endl;
                break;
            case 1: 
                palmTrace.clear(); 
                std::cout << "Palm trace cleared" << std::endl;
                break;
            case 2: 
                for (int i = 4; i >= 0; i--) {
                    if (fingerBasesMarked[i]) {
                        fingerBasesMarked[i] = false;
                        fingerBases[i] = cv::Point2f(-1, -1);
                        if (i == 0) thumbBase = cv::Point2f(-1, -1);
                        if (i == 4) pinkyBase = cv::Point2f(-1, -1);
                        std::cout << "Cleared " << fingerNames[i] << " base" << std::endl;
                        break;
                    }
                }
                break;
            case 3: 
                for (int i = 4; i >= 0; i--) {
                    if (fingerTipsMarked[i]) {
                        fingerTipsMarked[i] = false;
                        fingerTips[i] = cv::Point2f(-1, -1);
                        std::cout << "Cleared " << fingerNames[i] << " tip" << std::endl;
                        break;
                    }
                }
                break;
            case 4: 
                if (wristRightMarked) {
                    wristRightMarked = false;
                    wristRight = cv::Point2f(-1, -1);
                    std::cout << "Cleared right wrist boundary" << std::endl;
                } else if (wristLeftMarked) {
                    wristLeftMarked = false;
                    wristLeft = cv::Point2f(-1, -1);
                    std::cout << "Cleared left wrist boundary" << std::endl;
                }
                break;
        }
    }
    else if (key == 27) {
        isActive = false;
        calibrationData.isCalibrated = false;
        std::cout << "Calibration cancelled" << std::endl;
        return false;
    }
    
    return false;
}

void EnhancedManualCalibrator::extractHSVFromSamplingRect() {
    if (hsvSamplingRect.empty() || calibrationFrameHSV.empty()) return;
    
    cv::Rect safeRect = hsvSamplingRect & cv::Rect(0, 0, 
        calibrationFrameHSV.cols, calibrationFrameHSV.rows);
    
    if (safeRect.area() < 50) return;
    
    std::vector<float> hValues, sValues, vValues;
    
    for (int y = safeRect.y; y < safeRect.y + safeRect.height; y++) {
        for (int x = safeRect.x; x < safeRect.x + safeRect.width; x++) {
            cv::Vec3b hsv = calibrationFrameHSV.at<cv::Vec3b>(y, x);
            hValues.push_back(hsv[0]);
            sValues.push_back(hsv[1]);
            vValues.push_back(hsv[2]);
        }
    }
    
    if (hValues.size() < 20) return;
    
    std::sort(hValues.begin(), hValues.end());
    std::sort(sValues.begin(), sValues.end());
    std::sort(vValues.begin(), vValues.end());
    
    size_t lowerIdx = static_cast<size_t>(hValues.size() * 0.1f);
    size_t upperIdx = static_cast<size_t>(hValues.size() * 0.9f);
    
    calibrationData.calibratedHSVLower = cv::Scalar(
        hValues[lowerIdx] * 0.9f,
        sValues[lowerIdx] * 0.9f,
        vValues[lowerIdx] * 0.9f
    );
    
    calibrationData.calibratedHSVUpper = cv::Scalar(
        hValues[upperIdx] * 1.1f,
        sValues[upperIdx] * 1.1f,
        vValues[upperIdx] * 1.1f
    );
    
    for (int i = 0; i < 3; i++) {
        if (calibrationData.calibratedHSVLower[i] < 0) calibrationData.calibratedHSVLower[i] = 0;
        if (calibrationData.calibratedHSVUpper[i] > (i == 0 ? 180.0f : 255.0f)) 
            calibrationData.calibratedHSVUpper[i] = (i == 0 ? 180.0f : 255.0f);
    }
    
    calibrationData.hsvSamplingRect = hsvSamplingRect;
    
    std::cout << "HSV Range: " 
              << "H[" << calibrationData.calibratedHSVLower[0] << "-" << calibrationData.calibratedHSVUpper[0] << "] "
              << "S[" << calibrationData.calibratedHSVLower[1] << "-" << calibrationData.calibratedHSVUpper[1] << "] "
              << "V[" << calibrationData.calibratedHSVLower[2] << "-" << calibrationData.calibratedHSVUpper[2] << "]" 
              << std::endl;
}

void EnhancedManualCalibrator::calculateEnhancedRatios() {
    calibrationData.thumbBase = thumbBase;
    calibrationData.pinkyBase = pinkyBase;
    
    calibrationData.fingerBases.clear();
    calibrationData.fingerTips.clear();
    for (int i = 0; i < 5; i++) {
        if (fingerBasesMarked[i]) {
            calibrationData.fingerBases.push_back(fingerBases[i]);
        }
        if (fingerTipsMarked[i]) {
            calibrationData.fingerTips.push_back(fingerTips[i]);
        }
    }
    
    calibrationData.handContour = palmTrace;
    
    cv::Point2f palmSum(0, 0);
    int baseCount = 0;
    for (int i = 0; i < 5; i++) {
        if (fingerBasesMarked[i]) {
            palmSum += fingerBases[i];
            baseCount++;
        }
    }
    
    if (baseCount > 0) {
        calibrationData.palmCenter = palmSum * (1.0f / baseCount);
    } else {
        calibrationData.palmCenter = (thumbBase + pinkyBase) * 0.5f;
    }
    
    calibrationData.thumbBaseOffset = thumbBase - calibrationData.palmCenter;
    calibrationData.pinkyBaseOffset = pinkyBase - calibrationData.palmCenter;
    
    calibrationData.thumbPinkyBaseWidth = cv::norm(thumbBase - pinkyBase);
    
    float maxFingerLength = 0.0f;
    float totalFingerDistance = 0.0f;
    int fingerCount = 0;
    
    for (int i = 0; i < 5; i++) {
        if (fingerBasesMarked[i] && fingerTipsMarked[i]) {
            calibrationData.fingers[i].base = fingerBases[i];
            calibrationData.fingers[i].tip = fingerTips[i];
            calibrationData.fingers[i].tipVector = fingerTips[i] - fingerBases[i];
            calibrationData.fingers[i].baseOffset = fingerBases[i] - calibrationData.palmCenter;
            calibrationData.fingers[i].length = cv::norm(calibrationData.fingers[i].tipVector);
            calibrationData.fingers[i].angle = std::atan2(
                calibrationData.fingers[i].tipVector.y,
                calibrationData.fingers[i].tipVector.x
            );
            calibrationData.fingers[i].isCalibrated = true;
            
            maxFingerLength = std::max(maxFingerLength, calibrationData.fingers[i].length);
            totalFingerDistance += cv::norm(fingerTips[i] - calibrationData.palmCenter);
            fingerCount++;
            
            FingerIdentity identity;
            identity.id = i;
            identity.rawTip = fingerTips[i];
            identity.displayTip = fingerTips[i];
            identity.isDetected = true;
            identity.confidence = 1.0f;
            identity.isLocked = true;
            
            calibrationData.fingerIdentities.push_back(identity);
        }
    }
    
    if (fingerCount > 0) {
        calibrationData.ratios.avgFingerDistance = totalFingerDistance / fingerCount;
    }
    
    if (wristLeftMarked && wristRightMarked) {
        calibrationData.wrist.left = wristLeft;
        calibrationData.wrist.right = wristRight;
        calibrationData.wrist.leftOffset = wristLeft - calibrationData.palmCenter;
        calibrationData.wrist.rightOffset = wristRight - calibrationData.palmCenter;
        calibrationData.wrist.width = cv::norm(wristLeft - wristRight);
        calibrationData.calibratedWristWidth = calibrationData.wrist.width;
        
        cv::Point2f wristMid = (wristLeft + wristRight) * 0.5f;
        calibrationData.wrist.verticalOffset = wristMid.y - calibrationData.palmCenter.y;
        calibrationData.wrist.isCalibrated = true;
        
        calibrationData.calibratedWristDistance = cv::norm(wristMid - calibrationData.palmCenter);
        
        if (calibrationData.thumbPinkyBaseWidth > 0) {
            calibrationData.wristPalmDistanceRatio = calibrationData.calibratedWristDistance / calibrationData.thumbPinkyBaseWidth;
        }
    }
    
    if (wristLeftMarked && wristRightMarked) {
        calibrationData.ratios.thumbToWrist = cv::norm(thumbBase - wristLeft);
        calibrationData.ratios.pinkyToWrist = cv::norm(pinkyBase - wristRight);
        calibrationData.ratios.wristWidth = calibrationData.wrist.width;
    }
    
    float thumbDist = cv::norm(thumbBase - calibrationData.palmCenter);
    float pinkyDist = cv::norm(pinkyBase - calibrationData.palmCenter);
    calibrationData.ratios.palmRadius = (thumbDist + pinkyDist) * 0.5f;
    calibrationData.ratios.maxFingerLength = maxFingerLength;
    
    std::vector<cv::Point2f> allPoints;
    for (int i = 0; i < 5; i++) {
        if (fingerBasesMarked[i]) allPoints.push_back(fingerBases[i]);
        if (fingerTipsMarked[i]) allPoints.push_back(fingerTips[i]);
    }
    if (wristLeftMarked) allPoints.push_back(wristLeft);
    if (wristRightMarked) allPoints.push_back(wristRight);
    
    if (!allPoints.empty()) {
        float minX = allPoints[0].x, maxX = allPoints[0].x;
        float minY = allPoints[0].y, maxY = allPoints[0].y;
        
        for (const auto& p : allPoints) {
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        }
        
        calibrationData.ratios.handWidth = maxX - minX;
        calibrationData.ratios.handHeight = maxY - minY;
    }
    
    if (palmTrace.size() >= 5) {
        calibrationData.fittedEllipse = cv::fitEllipse(palmTrace);
    }
}

bool EnhancedManualCalibrator::finalizeCalibration() {
    for (int i = 0; i < 5; i++) {
        if (!validateStep(i)) {
            std::cout << "CALIBRATION FAILED: Step " << (i + 1) << " incomplete." << std::endl;
            return false;
        }
    }
    
    extractHSVFromSamplingRect();
    calculateEnhancedRatios();
    
    calibrationData.isCalibrated = true;
    isActive = false;
    isCalibrated = true;
    geometryUpdater.reset();
    shapeTracker.reset();
    
    std::cout << "\n=== CALIBRATION COMPLETE ===" << std::endl;
    std::cout << "Thumb-Pinky Base Width: " << calibrationData.thumbPinkyBaseWidth << std::endl;
    std::cout << "Avg Finger Distance: " << calibrationData.ratios.avgFingerDistance << std::endl;
    std::cout << "Wrist Distance: " << calibrationData.calibratedWristDistance << std::endl;
    std::cout << "Wrist Width: " << calibrationData.calibratedWristWidth << std::endl;
    std::cout << "Wrist-Palm Ratio: " << calibrationData.wristPalmDistanceRatio << std::endl;
    
    return true;
}