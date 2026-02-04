#include "TemporalSmoother.h"
#include <algorithm>
#include <cmath>

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

cv::Point2f TemporalSmoother::smoothWithVelocity(const cv::Point2f& current) {
    if (lastPosition.x < 0) {
        lastPosition = current;
        lastVelocity = cv::Point2f(0, 0);
        lastAcceleration = cv::Point2f(0, 0);
        positionHistory.clear();
        velocityHistory.clear();
        return current;
    }
    
    // Store history
    positionHistory.push_back(current);
    if (positionHistory.size() > 5) {
        positionHistory.pop_front();
    }
    
    // Compute velocity
    cv::Point2f velocity = current - lastPosition;
    velocityHistory.push_back(velocity);
    if (velocityHistory.size() > 3) {
        velocityHistory.pop_front();
    }
    
    // Smooth velocity
    cv::Point2f smoothedVelocity(0, 0);
    for (const auto& v : velocityHistory) {
        smoothedVelocity += v;
    }
    if (!velocityHistory.empty()) {
        smoothedVelocity *= (1.0f / velocityHistory.size());
    }
    
    // Compute acceleration
    cv::Point2f acceleration = smoothedVelocity - lastVelocity;
    
    // Limit jerk (change in acceleration)
    float jerk = cv::norm(acceleration - lastAcceleration);
    if (jerk > JERK_LIMIT && jerk > 0.001f) {
        acceleration = lastAcceleration + (acceleration - lastAcceleration) * (JERK_LIMIT / jerk);
    }
    
    // Predict next position with acceleration
    cv::Point2f predicted = lastPosition + smoothedVelocity + acceleration * 0.5f;
    
    // Adaptive smoothing based on velocity
    float velMag = cv::norm(smoothedVelocity);
    float alpha = computeAdaptiveAlpha(velMag);
    
    // Apply smoothing
    cv::Point2f smoothed = lastPosition * (1.0f - alpha) + predicted * alpha;
    
    // Update state
    lastPosition = smoothed;
    lastVelocity = smoothedVelocity;
    lastAcceleration = acceleration;
    
    return smoothed;
}

float TemporalSmoother::computeAdaptiveAlpha(float velocity) const {
    // Higher velocity = less smoothing (more responsive)
    // Lower velocity = more smoothing (more stable)
    
    if (velocity < VELOCITY_DEADZONE) {
        return MAX_SMOOTHING_ALPHA; // Very stable when still
    } else if (velocity > 50.0f) {
        return MIN_SMOOTHING_ALPHA; // Very responsive when moving fast
    } else {
        // Linear interpolation between min and max
        float ratio = (velocity - VELOCITY_DEADZONE) / (50.0f - VELOCITY_DEADZONE);
        return MAX_SMOOTHING_ALPHA - ratio * (MAX_SMOOTHING_ALPHA - MIN_SMOOTHING_ALPHA);
    }
}

cv::Point2f TemporalSmoother::smoothWithJerkLimit(const cv::Point2f& current) {
    if (lastPosition.x < 0) {
        lastPosition = current;
        return current;
    }
    
    cv::Point2f delta = current - lastPosition;
    float deltaLen = cv::norm(delta);
    
    // Limit maximum movement per frame
    if (deltaLen > 50.0f) { // Max 50 pixels per frame
        delta = delta * (50.0f / deltaLen);
    }
    
    cv::Point2f newPosition = lastPosition + delta;
    
    // Apply deadzone to eliminate micro-jitter
    cv::Point2f deadzoneDelta = applyDeadZone(newPosition - lastPosition, VELOCITY_DEADZONE);
    newPosition = lastPosition + deadzoneDelta;
    
    lastPosition = newPosition;
    return newPosition;
}

cv::Point2f TemporalSmoother::exponentialSmooth(const cv::Point2f& current, float alpha) {
    if (lastPosition.x < 0) {
        lastPosition = current;
        return current;
    }
    
    cv::Point2f smoothed = lastPosition * (1.0f - alpha) + current * alpha;
    lastPosition = smoothed;
    return smoothed;
}

void TemporalSmoother::reset() {
    lastPosition = cv::Point2f(-1, -1);
    lastVelocity = cv::Point2f(0, 0);
    lastAcceleration = cv::Point2f(0, 0);
    positionHistory.clear();
    velocityHistory.clear();
}