#include "ManualCalibrator.h"
#include <iostream>

extern CalibrationResult calibrationData;
extern std::atomic<bool> isCalibrated;
extern GeometryUpdater geometryUpdater;
extern ShapeAnchoredTracker shapeTracker;

void EnhancedManualCalibrator::startCalibration(const cv::Mat& frame) {
    frame.copyTo(calibrationFrame);
    cv::cvtColor(calibrationFrame, calibrationFrameHSV, cv::COLOR_BGR2HSV);
    
    hsvRectStart = cv::Point(-1, -1);
    hsvSamplingRect = cv::Rect();
    palmTrace.clear();
    
    // Initialize points
    wristMid = cv::Point2f(-1, -1);
    
    for (int i = 0; i < 5; i++) {
        fingerTips[i] = cv::Point2f(-1, -1);
        fingerBases[i] = cv::Point2f(-1, -1);
        fingerTipsMarked[i] = false;
        fingerBasesMarked[i] = false;
    }
    
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
                wristMid = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
                std::cout << "Wrist mid point marked" << std::endl;
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
        case 4: return wristMid.x >= 0;
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
            stepText = "STEP 5: Mark wrist MID point";
            instructionText = "Click wrist midpoint (between left and right wrist) - Press N when done";
            if (wristMid.x >= 0) {
                cv::circle(frame, wristMid, 10, cv::Scalar(0, 0, 255), -1);
                cv::putText(frame, "Wrist Mid", 
                           cv::Point(static_cast<int>(wristMid.x + 15), 
                                    static_cast<int>(wristMid.y)),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
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
                    case 4: std::cout << "Step 5: Mark wrist MID point" << std::endl; break;
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
                wristMid = cv::Point2f(-1, -1);
                std::cout << "Cleared wrist mid point" << std::endl;
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
    // Calculate palm center from finger bases
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
        // Fallback: average of thumb and pinky bases if available
        cv::Point2f thumbBase = fingerBasesMarked[0] ? fingerBases[0] : cv::Point2f(-1, -1);
        cv::Point2f pinkyBase = fingerBasesMarked[4] ? fingerBases[4] : cv::Point2f(-1, -1);
        if (thumbBase.x >= 0 && pinkyBase.x >= 0) {
            calibrationData.palmCenter = (thumbBase + pinkyBase) * 0.5f;
        }
    }
    
    // Store wrist mid point
    if (wristMid.x >= 0) {
        calibrationData.wrist.left = cv::Point2f(wristMid.x - 20, wristMid.y);
        calibrationData.wrist.right = cv::Point2f(wristMid.x + 20, wristMid.y);
        calibrationData.wrist.width = 40.0f;
        calibrationData.wrist.isCalibrated = true;
    }
    
    // Store finger data
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
    calibrationData.isCalibrated = true;
    
    std::cout << "\n=== CALIBRATION COMPLETE ===" << std::endl;
    std::cout << "Palm center: (" << calibrationData.palmCenter.x << ", " << calibrationData.palmCenter.y << ")" << std::endl;
    std::cout << "Wrist mid: (" << wristMid.x << ", " << wristMid.y << ")" << std::endl;
    std::cout << "Fingers calibrated: " << calibrationData.fingerTips.size() << std::endl;
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
    
    return true;
}