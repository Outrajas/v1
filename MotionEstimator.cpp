#include "MotionEstimator.h"
#include <algorithm>

constexpr float CURSOR_MAX_DELTA_PER_FRAME = 50.0f;
constexpr float CURSOR_VELOCITY_SMOOTHING_ALPHA = 0.3f;
constexpr float CURSOR_FAST_THRESHOLD = 30.0f;

cv::Mat MotionEstimator::computeSmoothedDelta(const cv::Mat& currentGray) {
    cv::Mat delta;
    
    if (!prevGray.empty()) {
        cv::absdiff(currentGray, prevGray, delta);
        
        deltaHistory.push_back(delta.clone());
        if (deltaHistory.size() > DELTA_SMOOTHING_HISTORY)
            deltaHistory.pop_front();
        
        if (deltaHistory.size() > 1) {
            cv::Mat smoothedDelta = cv::Mat::zeros(delta.size(), delta.type());
            
            float totalWeight = 0;
            for (size_t i = 0; i < deltaHistory.size(); i++) {
                float weight = static_cast<float>(i + 1) / deltaHistory.size();
                cv::addWeighted(smoothedDelta, 1.0, deltaHistory[i], weight, 0, smoothedDelta);
                totalWeight += weight;
            }
            
            if (totalWeight > 0) {
                delta = smoothedDelta * (1.0f / totalWeight);
            }
        }
        
        cv::threshold(delta, delta, DELTA_THRESHOLD, 255, cv::THRESH_BINARY);
    } else {
        delta = cv::Mat::zeros(currentGray.size(), CV_8UC1);
    }
    
    currentGray.copyTo(prevGray);
    return delta;
}

void MotionEstimator::getMotionMask(const cv::Mat& frame, cv::Mat& motionMaskOut, float& deltaStrength) {
    if (skipCounter < 2) {
        skipCounter++;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        prevGray = gray.clone();
        motionMaskOut = cv::Mat::zeros(frame.size(), CV_8UC1);
        deltaStrength = 0.0f;
        return;
    }
    
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    motionMaskOut = computeSmoothedDelta(gray);
    
    if (!motionMaskOut.empty()) {
        deltaStrength = cv::mean(motionMaskOut)[0] / 255.0f;
        deltaStrength = std::min(1.0f, deltaStrength * 2.0f); // Scale for better sensitivity
    }
}