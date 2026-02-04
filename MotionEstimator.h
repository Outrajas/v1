#ifndef MOTION_ESTIMATOR_H
#define MOTION_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <deque>

constexpr int DELTA_THRESHOLD = 15;
constexpr int DELTA_SMOOTHING_HISTORY = 3;

class MotionEstimator {
private:
    cv::Mat prevGray;
    std::deque<cv::Mat> deltaHistory;
    int skipCounter = 0;
    
    cv::Mat computeSmoothedDelta(const cv::Mat& currentGray);
    
public:
    MotionEstimator() = default;
    
    void getMotionMask(const cv::Mat& frame, cv::Mat& motionMaskOut, float& deltaStrength);
};

#endif