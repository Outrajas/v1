#ifndef TEMPORAL_SMOOTHER_H
#define TEMPORAL_SMOOTHER_H

#include <opencv2/opencv.hpp>

class TemporalSmoother {
public:
    TemporalSmoother() = default;
    
    cv::Point2f smoothPoint(const cv::Point2f& current, const cv::Point2f& previous, float alpha);
    float smoothValue(float current, float previous, float alpha);
    cv::Point2f clampPoint(const cv::Point2f& point, const cv::Point2f& center, float maxDistance);
    
    cv::Point2f computeVelocity(const cv::Point2f& current, const cv::Point2f& previous);
    float computeVelocityMagnitude(const cv::Point2f& current, const cv::Point2f& previous);
    
    cv::Point2f applyDeadZone(const cv::Point2f& delta, float deadZone);
    float applyDeadZone(float value, float deadZone);
};

#endif