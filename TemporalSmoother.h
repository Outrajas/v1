#ifndef TEMPORAL_SMOOTHER_H
#define TEMPORAL_SMOOTHER_H

#include <opencv2/opencv.hpp>
#include <deque>
#include <algorithm>

constexpr float JERK_LIMIT = 100.0f;      // Maximum acceleration change per frame
constexpr float VELOCITY_DEADZONE = 2.0f; // Minimum movement to register
constexpr float MAX_SMOOTHING_ALPHA = 0.8f;
constexpr float MIN_SMOOTHING_ALPHA = 0.2f;

class TemporalSmoother {
private:
    cv::Point2f lastPosition;
    cv::Point2f lastVelocity;
    cv::Point2f lastAcceleration;
    std::deque<cv::Point2f> positionHistory;
    std::deque<cv::Point2f> velocityHistory;
    
public:
    TemporalSmoother() = default;
    
    cv::Point2f smoothPoint(const cv::Point2f& current, const cv::Point2f& previous, float alpha);
    float smoothValue(float current, float previous, float alpha);
    cv::Point2f clampPoint(const cv::Point2f& point, const cv::Point2f& center, float maxDistance);
    
    cv::Point2f computeVelocity(const cv::Point2f& current, const cv::Point2f& previous);
    float computeVelocityMagnitude(const cv::Point2f& current, const cv::Point2f& previous);
    
    cv::Point2f applyDeadZone(const cv::Point2f& delta, float deadZone);
    float applyDeadZone(float value, float deadZone);
    
    // New: Advanced smoothing with velocity awareness
    cv::Point2f smoothWithVelocity(const cv::Point2f& current);
    float computeAdaptiveAlpha(float velocity) const;
    
    // New: Jerk-limited smoothing (prevents sudden changes)
    cv::Point2f smoothWithJerkLimit(const cv::Point2f& current);
    
    // New: Exponential moving average with variable window
    cv::Point2f exponentialSmooth(const cv::Point2f& current, float alpha);
    
    // New: Reset history
    void reset();
};

#endif