#ifndef MANUAL_CALIBRATOR_H
#define MANUAL_CALIBRATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "FingerAnalyzer.h"

class EnhancedManualCalibrator {
private:
    cv::Mat calibrationFrame;
    cv::Mat calibrationFrameHSV;
    
    int calibrationStep = 0;
    bool isActive = false;
    bool isDrawing = false;
    
    cv::Point hsvRectStart;
    cv::Rect hsvSamplingRect;
    std::vector<cv::Point> palmTrace;
    
    // TWO-POINT WRIST MODEL - No single midpoint
    cv::Point2f wristLeft;
    cv::Point2f wristRight;
    bool wristLeftMarked = false;
    bool wristRightMarked = false;
    
    cv::Point2f fingerTips[5];
    cv::Point2f fingerBases[5];
    bool fingerTipsMarked[5] = {false};
    bool fingerBasesMarked[5] = {false};
    
    std::string fingerNames[5] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
    
public:
    void startCalibration(const cv::Mat& frame);
    void handleMouse(int event, int x, int y);
    bool validateStep(int step) const;
    void drawStepUI(cv::Mat& frame);
    bool processCalibration(int key, cv::Mat& displayFrame);
    void extractHSVFromSamplingRect();
    void calculateEnhancedRatios();
    bool finalizeCalibration();
    bool isCalibrating() const { return isActive; }
    
private:
    // Helper functions for geometric constraints
    cv::Point2f projectToContour(const cv::Point2f& point) const;
    bool validateWristGeometry() const;
    void normalizeWristPoints();
    bool isWristOppositeFingers() const;
    bool isWristWidthValid() const;
};

#endif