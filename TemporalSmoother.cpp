#include "TemporalSmoother.h"
#include <algorithm>

cv::Point2f TemporalSmoother::smoothPoint(const cv::Point2f& current, const cv::Point2f& previous, float alpha) {
    if (previous.x < 0) return current;
    return previous * (1.0f - alpha) + current * alpha;
}

float TemporalSmoother::smoothValue(float current, float previous, float alpha) {
    if (previous < 0) return current;
    return previous * (1.0f - alpha) + current * alpha;
}

cv::Point2f TemporalSmoother::clampPoint(const cv::Point2f& point, const cv::Point2f& center, float maxDistance) {
    cv::Point2f delta = point - center;
    float distance = cv::norm(delta);
    if (distance <= maxDistance) return point;
    
    cv::Point2f direction = delta * (1.0f / distance);
    return center + direction * maxDistance;
}

cv::Point2f TemporalSmoother::computeVelocity(const cv::Point2f& current, const cv::Point2f& previous) {
    if (previous.x < 0) return cv::Point2f(0, 0);
    return current - previous;
}

float TemporalSmoother::computeVelocityMagnitude(const cv::Point2f& current, const cv::Point2f& previous) {
    if (previous.x < 0) return 0.0f;
    return cv::norm(current - previous);
}

cv::Point2f TemporalSmoother::applyDeadZone(const cv::Point2f& delta, float deadZone) {
    float magnitude = cv::norm(delta);
    if (magnitude < deadZone) return cv::Point2f(0, 0);
    return delta;
}

float TemporalSmoother::applyDeadZone(float value, float deadZone) {
    if (std::abs(value) < deadZone) return 0.0f;
    return value;
}