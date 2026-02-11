// PalmEstimator.cpp
#include "PalmEstimator.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>

extern cv::Point2f lastValidPalm;
extern cv::Point2f lastValidThumbBase;
extern cv::Point2f lastValidPinkyBase;
extern int failureFrameCount;
extern std::atomic<bool> isCalibrated;
extern CalibrationResult calibrationData;
extern GeometryUpdater geometryUpdater;
extern ShapeAnchoredTracker shapeTracker;

PalmEstimator::PalmEstimator() 
    : m_lastSmoothedWristLeft(-1, -1)
    , m_lastSmoothedWristRight(-1, -1)
    , m_lastStableWristLeft(-1, -1)
    , m_lastStableWristRight(-1, -1)
    , m_lastSmoothedPalm(-1, -1)
    , m_lastAngle(0.0f)
    , m_lastStableAngle(0.0f)
    , m_stableFrameCount(0)
    , m_lastAngularVelocity(0.0f)
    , m_hasInitialState(false)
    , m_currentFreezeState(FREEZE_STATE_FREE)
    , m_freezeCounter(0)
    , m_unfreezeCounter(0)
    , m_freezeActive(false)
    , m_frozenWristLeft(-1, -1)
    , m_frozenWristRight(-1, -1)
    , m_frozenWristMid(-1, -1)
    , m_lastRawWristLeft(-1, -1)
    , m_lastRawWristRight(-1, -1)
{
    m_palmSmoothingBuffer.clear();
}

bool PalmEstimator::isPointInContour(const cv::Point2f& point, const std::vector<cv::Point>& contour) const {
    if (contour.empty()) return true;
    cv::Point point_int(static_cast<int>(point.x), static_cast<int>(point.y));
    return cv::pointPolygonTest(contour, point_int, false) >= 0;
}

cv::Point2f PalmEstimator::projectPointToContourInterior(const cv::Point2f& point, 
                                                         const std::vector<cv::Point>& contour,
                                                         const cv::Point2f& fallback) const {
    if (contour.empty()) return point;
    
    if (isPointInContour(point, contour)) {
        return point;
    }
    
    cv::Point2f nearest = point;
    float minDist = std::numeric_limits<float>::max();
    
    for (const auto& p : contour) {
        cv::Point2f contourPoint(p);
        float dist = cv::norm(contourPoint - point);
        if (dist < minDist) {
            minDist = dist;
            nearest = contourPoint;
        }
    }
    
    return nearest;
}

float PalmEstimator::computePalmCoreConfidence(const std::vector<cv::Point>& contour,
                                              const cv::Point2f& palmCenter,
                                              float palmRadius) const {
    if (contour.empty() || palmRadius < 0.001f) return 0.5f;
    
    std::vector<float> distances;
    distances.reserve(contour.size());
    
    for (const auto& p : contour) {
        distances.push_back(cv::norm(cv::Point2f(p) - palmCenter));
    }
    
    size_t n = distances.size() / 2;
    std::nth_element(distances.begin(), distances.begin() + n, distances.end());
    float medianDist = distances[n];
    
    float threshold = 1.2f * medianDist;
    int countInCore = 0;
    
    for (float d : distances) {
        if (d <= threshold) countInCore++;
    }
    
    return static_cast<float>(countInCore) / contour.size();
}

float PalmEstimator::computeWristSpanConfidence(float wristWidth, float thumbPinkyWidth) const {
    if (thumbPinkyWidth < 0.001f) return 0.5f;
    
    float ratio = wristWidth / thumbPinkyWidth;
    
    if (ratio >= 0.5f && ratio <= 1.2f) {
        return 1.0f;
    } else if (ratio < 0.5f) {
        return std::max(0.0f, 1.0f - (0.5f - ratio) / 0.5f);
    } else {
        return std::max(0.0f, 1.0f - (ratio - 1.2f) / 0.3f);
    }
}

float PalmEstimator::computeContourCompactness(const std::vector<cv::Point>& contour) const {
    if (contour.size() < 3) return 0.5f;
    
    double contourArea = cv::contourArea(contour);
    
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double hullArea = cv::contourArea(hull);
    
    if (hullArea < 0.001f) return 0.5f;
    
    float compactness = static_cast<float>(contourArea / hullArea);
    return std::max(0.0f, std::min(1.0f, compactness));
}

float PalmEstimator::computeArmIntrusionConfidence(const std::vector<cv::Point>& contour,
                                                  const cv::Point2f& palmCenter,
                                                  const cv::Point2f& wristMid,
                                                  float palmRadius) const {
    if (contour.empty() || palmRadius < 0.001f) return 0.5f;
    
    cv::Point2f yAxis = wristMid - palmCenter;
    float axisLength = cv::norm(yAxis);
    
    if (axisLength < 0.001f) {
        yAxis = cv::Point2f(0, 1);
        axisLength = 1.0f;
    } else {
        yAxis = yAxis * (1.0f / axisLength);
    }
    
    int armSidePoints = 0;
    
    for (const auto& p : contour) {
        cv::Point2f vec = cv::Point2f(p) - palmCenter;
        float localY = vec.dot(yAxis);
        if (localY > 0.6f * palmRadius) {
            armSidePoints++;
        }
    }
    
    float confidence = 1.0f - static_cast<float>(armSidePoints) / contour.size();
    return std::max(0.0f, std::min(1.0f, confidence));
}

float PalmEstimator::computeTemporalConfidence(const cv::Point2f& currentRawLeft,
                                              const cv::Point2f& currentRawRight,
                                              const cv::Point2f& lastRawLeft,
                                              const cv::Point2f& lastRawRight) const {
    if (lastRawLeft.x < 0 || lastRawRight.x < 0) {
        return 1.0f;
    }
    
    cv::Point2f currentMid = (currentRawLeft + currentRawRight) * 0.5f;
    cv::Point2f lastMid = (lastRawLeft + lastRawRight) * 0.5f;
    
    float velocity = cv::norm(currentMid - lastMid);
    float confidence = 1.0f - std::min(1.0f, velocity / VELOCITY_CAP);
    
    return std::max(0.0f, std::min(1.0f, confidence));
}

PalmEstimator::StructuralConfidence PalmEstimator::computeStructuralConfidence(
    const std::vector<cv::Point>& contour,
    const cv::Point2f& palmCenter,
    const cv::Point2f& wristLeft,
    const cv::Point2f& wristRight,
    const cv::Point2f& wristMid,
    const cv::Point2f& thumbBase,
    const cv::Point2f& pinkyBase,
    float palmRadius,
    const cv::Point2f& lastRawWristLeft,
    const cv::Point2f& lastRawWristRight) {
    
    StructuralConfidence conf;
    
    float thumbPinkyWidth = cv::norm(pinkyBase - thumbBase);
    float wristWidth = cv::norm(wristRight - wristLeft);
    
    conf.palmCoreConfidence = computePalmCoreConfidence(contour, palmCenter, palmRadius);
    conf.wristSpanConfidence = computeWristSpanConfidence(wristWidth, thumbPinkyWidth);
    conf.contourCompactness = computeContourCompactness(contour);
    conf.armIntrusionConfidence = computeArmIntrusionConfidence(contour, palmCenter, wristMid, palmRadius);
    conf.temporalConfidence = computeTemporalConfidence(wristLeft, wristRight, 
                                                       lastRawWristLeft, lastRawWristRight);
    
    // NEW: Temporal confidence weighted equally with other components
    conf.globalScore = 
        0.25f * conf.palmCoreConfidence +
        0.20f * conf.wristSpanConfidence +
        0.15f * conf.contourCompactness +
        0.20f * conf.armIntrusionConfidence +
        0.20f * conf.temporalConfidence;
    
    conf.globalScore = std::max(0.0f, std::min(1.0f, conf.globalScore));
    
    return conf;
}

cv::Point2f PalmEstimator::smoothPalm(const cv::Point2f& rawPalm) {
    if (rawPalm.x < 0) {
        return rawPalm;
    }
    
    m_palmSmoothingBuffer.push_back(rawPalm);
    if (m_palmSmoothingBuffer.size() > 5) {
        m_palmSmoothingBuffer.pop_front();
    }
    
    cv::Point2f smoothed(0, 0);
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < m_palmSmoothingBuffer.size(); i++) {
        float weight = static_cast<float>(i + 1) / m_palmSmoothingBuffer.size();
        smoothed += m_palmSmoothingBuffer[i] * weight;
        totalWeight += weight;
    }
    
    if (totalWeight > 0.0f) {
        smoothed = smoothed * (1.0f / totalWeight);
    } else {
        smoothed = rawPalm;
    }
    
    m_lastSmoothedPalm = smoothed;
    return smoothed;
}

void PalmEstimator::updateFreezeState(float globalScore, Result& result) {
    // State transition logic with hysteresis
    switch (m_currentFreezeState) {
        case FREEZE_STATE_FREE:
            if (globalScore < FREEZE_CONFIDENCE_EXIT_THRESHOLD) {
                m_freezeCounter++;
                if (m_freezeCounter >= FREEZE_FRAMES_REQUIRED) {
                    m_currentFreezeState = FREEZE_STATE_FROZEN;
                    m_freezeActive = true;
                    m_freezeCounter = 0;
                    m_unfreezeCounter = 0;
                    
                    // Capture frozen position
                    m_frozenWristLeft = m_lastSmoothedWristLeft;
                    m_frozenWristRight = m_lastSmoothedWristRight;
                    m_frozenWristMid = (m_frozenWristLeft + m_frozenWristRight) * 0.5f;
                }
            } else {
                m_freezeCounter = 0;
            }
            
            if (globalScore < FREEZE_CONFIDENCE_ENTRY_THRESHOLD && 
                globalScore >= FREEZE_CONFIDENCE_EXIT_THRESHOLD) {
                m_currentFreezeState = FREEZE_STATE_DAMPED;
            }
            break;
            
        case FREEZE_STATE_DAMPED:
            if (globalScore >= FREEZE_CONFIDENCE_ENTRY_THRESHOLD) {
                m_currentFreezeState = FREEZE_STATE_FREE;
                m_freezeCounter = 0;
                m_unfreezeCounter = 0;
            } else if (globalScore < FREEZE_CONFIDENCE_EXIT_THRESHOLD) {
                m_freezeCounter++;
                if (m_freezeCounter >= FREEZE_FRAMES_REQUIRED) {
                    m_currentFreezeState = FREEZE_STATE_FROZEN;
                    m_freezeActive = true;
                    m_freezeCounter = 0;
                    m_unfreezeCounter = 0;
                    
                    m_frozenWristLeft = m_lastSmoothedWristLeft;
                    m_frozenWristRight = m_lastSmoothedWristRight;
                    m_frozenWristMid = (m_frozenWristLeft + m_frozenWristRight) * 0.5f;
                }
            }
            break;
            
        case FREEZE_STATE_FROZEN:
            if (globalScore >= FREEZE_CONFIDENCE_ENTRY_THRESHOLD) {
                m_unfreezeCounter++;
                if (m_unfreezeCounter >= UNFREEZE_FRAMES_REQUIRED) {
                    m_currentFreezeState = FREEZE_STATE_FREE;
                    m_freezeActive = false;
                    m_freezeCounter = 0;
                    m_unfreezeCounter = 0;
                }
            } else {
                m_unfreezeCounter = 0;
            }
            break;
    }
    
    // Update result state
    result.freezeState = m_currentFreezeState;
    result.freezeCounter = (m_currentFreezeState == FREEZE_STATE_FROZEN) ? 
                           m_unfreezeCounter : m_freezeCounter;
    result.unfreezeCounter = m_unfreezeCounter;
}

void PalmEstimator::applyExponentialSmoothing(cv::Point2f& smoothedLeft,
                                             cv::Point2f& smoothedRight,
                                             const cv::Point2f& rawLeft,
                                             const cv::Point2f& rawRight,
                                             float alpha,
                                             bool skipSmoothing) {
    if (!m_hasInitialState) {
        smoothedLeft = rawLeft;
        smoothedRight = rawRight;
        m_lastSmoothedWristLeft = rawLeft;
        m_lastSmoothedWristRight = rawRight;
        m_hasInitialState = true;
    } else if (skipSmoothing) {
        smoothedLeft = m_lastSmoothedWristLeft;
        smoothedRight = m_lastSmoothedWristRight;
    } else {
        smoothedLeft = m_lastSmoothedWristLeft * (1.0f - alpha) + rawLeft * alpha;
        smoothedRight = m_lastSmoothedWristRight * (1.0f - alpha) + rawRight * alpha;
        m_lastSmoothedWristLeft = smoothedLeft;
        m_lastSmoothedWristRight = smoothedRight;
    }
}

void PalmEstimator::applyProportionalDamping(cv::Point2f& left,
                                            cv::Point2f& right,
                                            float globalScore) {
    if (globalScore < 0.7f && m_lastStableWristLeft.x >= 0 && m_lastStableWristRight.x >= 0) {
        float dampingFactor = std::max(0.0f, std::min(0.6f, 1.0f - globalScore));
        float blend = dampingFactor * 0.4f;
        
        left = left * (1.0f - blend) + m_lastStableWristLeft * blend;
        right = right * (1.0f - blend) + m_lastStableWristRight * blend;
        
        m_lastSmoothedWristLeft = left;
        m_lastSmoothedWristRight = right;
    }
}

void PalmEstimator::updateStableAnchors(const cv::Point2f& left,
                                       const cv::Point2f& right,
                                       float globalScore) {
    if (globalScore > 0.7f) {
        m_lastStableWristLeft = left;
        m_lastStableWristRight = right;
    }
}

std::pair<int, int> PalmEstimator::findThumbPinkyDefects(const std::vector<cv::Point>& contour,
                                                        const std::vector<cv::Vec4i>& defects) const {
    if (defects.size() < 2) {
        if (defects.size() >= 2) {
            return {0, 1};
        } else if (defects.size() == 1) {
            return {0, 0};
        }
        return {-1, -1};
    }
    
    std::vector<std::pair<float, int>> defectDepths;
    for (size_t i = 0; i < defects.size(); i++) {
        float depth = defects[i][3] / 256.0f;
        defectDepths.emplace_back(depth, static_cast<int>(i));
    }
    
    std::sort(defectDepths.begin(), defectDepths.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    if (defectDepths.size() < 2) {
        return {defectDepths[0].second, defectDepths[0].second};
    }
    
    int deepestIdx = defectDepths[0].second;
    int secondDeepestIdx = defectDepths[1].second;
    
    cv::Point2f deepestPoint(contour[defects[deepestIdx][2]]);
    cv::Point2f secondPoint(contour[defects[secondDeepestIdx][2]]);
    
    if (deepestPoint.x < secondPoint.x) {
        return {deepestIdx, secondDeepestIdx};
    } else {
        return {secondDeepestIdx, deepestIdx};
    }
}

float PalmEstimator::computeWristGeometry(const std::vector<cv::Point>& rawContour,
                                        cv::Point2f& wristLeft,
                                        cv::Point2f& wristRight,
                                        cv::Point2f& wristMid) {
    wristLeft = cv::Point2f(-1, -1);
    wristRight = cv::Point2f(-1, -1);
    wristMid = cv::Point2f(-1, -1);
    
    float confidence = 0.0f;
    
    if (rawContour.size() < 20) {
        return confidence;
    }
    
    std::vector<int> hullIndices;
    cv::convexHull(rawContour, hullIndices, false, false);
    
    std::vector<cv::Vec4i> defects;
    if (hullIndices.size() > 3) {
        cv::convexityDefects(rawContour, hullIndices, defects);
    }
    
    if (defects.empty()) {
        cv::Rect bbox = cv::boundingRect(rawContour);
        wristLeft = cv::Point2f(bbox.x + bbox.width * 0.2f, bbox.y + bbox.height * 0.9f);
        wristRight = cv::Point2f(bbox.x + bbox.width * 0.8f, bbox.y + bbox.height * 0.9f);
        wristMid = (wristLeft + wristRight) * 0.5f;
        confidence = 0.1f;
        return confidence;
    }
    
    auto [thumbDefectIdx, pinkyDefectIdx] = findThumbPinkyDefects(rawContour, defects);
    
    if (thumbDefectIdx < 0 || pinkyDefectIdx < 0) {
        if (!defects.empty()) {
            thumbDefectIdx = 0;
            pinkyDefectIdx = std::min(1, (int)defects.size() - 1);
            confidence = 0.2f;
        } else {
            cv::Rect bbox = cv::boundingRect(rawContour);
            wristLeft = cv::Point2f(bbox.x + bbox.width * 0.25f, bbox.y + bbox.height * 0.85f);
            wristRight = cv::Point2f(bbox.x + bbox.width * 0.75f, bbox.y + bbox.height * 0.85f);
            wristMid = (wristLeft + wristRight) * 0.5f;
            confidence = 0.1f;
            return confidence;
        }
    }
    
    const auto& thumbDefect = defects[thumbDefectIdx];
    const auto& pinkyDefect = defects[pinkyDefectIdx];
    
    cv::Point2f thumbBase(rawContour[thumbDefect[2]]);
    cv::Point2f pinkyBase(rawContour[pinkyDefect[2]]);
    
    cv::Point2f handAxis = pinkyBase - thumbBase;
    float axisLength = cv::norm(handAxis);
    
    if (axisLength < 30.0f) {
        float scale = 30.0f / std::max(axisLength, 1.0f);
        handAxis = handAxis * scale;
        axisLength = 30.0f;
        confidence = 0.3f;
    } else {
        confidence = 0.5f;
    }
    
    cv::Point2f axisDir = handAxis / axisLength;
    cv::Point2f perpendicular(-axisDir.y, axisDir.x);
    
    cv::Point2f roughPalmCenter = (thumbBase + pinkyBase) * 0.5f;
    
    float wristDistance = axisLength * 0.8f;
    wristMid = roughPalmCenter - perpendicular * wristDistance;
    float wristHalfWidth = axisLength * 0.3f;
    
    wristLeft = wristMid - axisDir * wristHalfWidth;
    wristRight = wristMid + axisDir * wristHalfWidth;
    
    if (isPointInContour(wristLeft, rawContour) && isPointInContour(wristRight, rawContour)) {
        confidence = std::min(1.0f, confidence + 0.3f);
    }
    
    return confidence;
}

PalmEstimator::Result PalmEstimator::detect(const cv::Mat& frame, const cv::Mat& motionMask, 
                                           const cv::Mat& skinMask, float hsvConfidence) {
    Result result;
    result.status = "Processing";
    result.wristHint = cv::Point2f(-1, -1);
    
    try {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(skinMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        result.contoursFound = contours.size();
        
        if (contours.empty()) {
            result.handDetected = false;
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
                    
                    contourCenter = projectPointToContourInterior(contourCenter, contours[i], contourCenter);
                    
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
            size_t largestIdx = 0;
            double largestArea = 0;
            for (size_t i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > largestArea) {
                    largestArea = area;
                    largestIdx = i;
                }
            }
            result.contour = contours[largestIdx];
            result.handDetected = true;
            result.status = "Hand (fallback contour)";
        } else {
            auto maxScoreIt = std::max_element(potentialScores.begin(), potentialScores.end());
            size_t bestIdxInPotentials = std::distance(potentialScores.begin(), maxScoreIt);
            size_t bestContourIdx = potentialIndices[bestIdxInPotentials];
            
            result.contour = contours[bestContourIdx];
            result.handDetected = true;
        }
        
        if (!result.contour.empty()) {
            result.area = cv::contourArea(result.contour);
            result.boundingBox = cv::boundingRect(result.contour);
            
            float aspect, solidity;
            couldBeHand(result.contour, aspect, solidity);
            result.aspectRatio = aspect;
            result.solidity = solidity;
            
            // ========== PALM CENTER COMPUTATION ==========
            std::vector<cv::Point> contourForPalm = result.contour;
            bool palmComputed = fitContourToModel(contourForPalm, result.palm, result.handSizeScale);
            
            // Apply palm smoothing
            result.palm = smoothPalm(result.palm);
            
            // CRITICAL: Assign contourForPalm as the authoritative contour for this frame
            result.contour = contourForPalm;
            
            // ========== RAW WRIST GEOMETRY ==========
            cv::Point2f rawWristLeft, rawWristRight, rawWristMid;
            float wristConfidence = computeWristGeometry(result.contour,
                                                        rawWristLeft,
                                                        rawWristRight,
                                                        rawWristMid);
            
            // Store raw wrist for confidence and overlay
            result.rawWristLeft = rawWristLeft;
            result.rawWristRight = rawWristRight;
            
            // ========== THUMB/PINKY BASES (FOR WRIST SPAN CONFIDENCE) ==========
            std::vector<int> hullIndices;
            cv::convexHull(result.contour, hullIndices, false, false);
            std::vector<cv::Vec4i> defects;
            if (hullIndices.size() > 3) {
                cv::convexityDefects(result.contour, hullIndices, defects);
            }
            
            cv::Point2f thumbBase(-1, -1), pinkyBase(-1, -1);
            if (!defects.empty()) {
                auto [thumbIdx, pinkyIdx] = findThumbPinkyDefects(result.contour, defects);
                if (thumbIdx >= 0 && pinkyIdx >= 0) {
                    thumbBase = cv::Point2f(result.contour[defects[thumbIdx][2]]);
                    pinkyBase = cv::Point2f(result.contour[defects[pinkyIdx][2]]);
                }
            }
            
            // ========== STRUCTURAL CONFIDENCE ==========
            float estimatedPalmRadius = std::sqrt(result.area / M_PI);
            StructuralConfidence structConf = computeStructuralConfidence(
                result.contour,
                result.palm,
                rawWristLeft,
                rawWristRight,
                rawWristMid,
                thumbBase,
                pinkyBase,
                estimatedPalmRadius,
                m_lastRawWristLeft,
                m_lastRawWristRight);
            
            // Store current raw wrist for next frame
            m_lastRawWristLeft = rawWristLeft;
            m_lastRawWristRight = rawWristRight;
            
            // ========== HYSTERESIS STATE MACHINE ==========
            updateFreezeState(structConf.globalScore, result);
            result.confidence = structConf.globalScore;
            result.temporalConfidence = structConf.temporalConfidence;
            
            // ========== CONTOUR SCULPTING ==========
            // ALWAYS use raw wrist for sculpting (invariant)
            result.constrainedContour = constrainContourWithWrist(result.contour,
                                                                rawWristLeft,
                                                                rawWristRight,
                                                                rawWristMid,
                                                                wristConfidence);
            
            // ========== ANGLE & ANGULAR VELOCITY ==========
            float currentAngle = 0.0f;
            if (result.palm.x >= 0 && rawWristMid.x >= 0) {
                currentAngle = std::atan2(rawWristMid.y - result.palm.y,
                                         rawWristMid.x - result.palm.x);
            }
            
            float deltaAngle = 0.0f;
            if (m_hasInitialState) {
                deltaAngle = std::atan2(std::sin(currentAngle - m_lastAngle),
                                       std::cos(currentAngle - m_lastAngle));
            }
            
            float deltaFromStable = 0.0f;
            if (m_hasInitialState) {
                deltaFromStable = std::atan2(std::sin(currentAngle - m_lastStableAngle),
                                            std::cos(currentAngle - m_lastStableAngle));
            }
            
            // ========== STABILITY DETECTION ==========
            if (std::abs(deltaFromStable) < m_stableAngleThreshold && structConf.globalScore > 0.6f) {
                m_stableFrameCount++;
            } else {
                m_stableFrameCount = 0;
                m_lastStableAngle = currentAngle;
            }
            
            // ========== ADAPTIVE ALPHA (ONLY IN FREE STATE) ==========
            float alpha = m_baseAlpha;
            bool skipSmoothing = false;
            
            switch (m_currentFreezeState) {
                case FREEZE_STATE_FREE:
                    alpha = m_baseAlpha + (structConf.globalScore - 0.5f) * ADAPTIVE_ALPHA_RANGE;
                    alpha = std::max(0.05f, std::min(m_maxAlpha, alpha));
                    
                    if (std::abs(deltaAngle) < m_idleVelocityThreshold && structConf.globalScore > 0.6f) {
                        alpha = std::min(alpha * m_idleAlphaBoost, m_maxAlpha);
                    }
                    skipSmoothing = false;
                    break;
                    
                case FREEZE_STATE_DAMPED:
                    alpha = DAMPED_ALPHA;
                    skipSmoothing = false;
                    break;
                    
                case FREEZE_STATE_FROZEN:
                    alpha = 0.0f;
                    skipSmoothing = true;
                    break;
            }
            
            result.effectiveAlpha = alpha;
            
            // ========== EXPONENTIAL SMOOTHING ==========
            cv::Point2f smoothedLeft = rawWristLeft;
            cv::Point2f smoothedRight = rawWristRight;
            
            if (m_currentFreezeState == FREEZE_STATE_FROZEN) {
                // Use frozen position
                smoothedLeft = m_frozenWristLeft;
                smoothedRight = m_frozenWristRight;
                result.wristLeft = smoothedLeft;
                result.wristRight = smoothedRight;
                result.wristMid = (smoothedLeft + smoothedRight) * 0.5f;
            } else {
                // Apply exponential smoothing
                applyExponentialSmoothing(smoothedLeft, smoothedRight,
                                         rawWristLeft, rawWristRight,
                                         alpha, skipSmoothing);
                
                // ========== PROPORTIONAL DAMPING ==========
                applyProportionalDamping(smoothedLeft, smoothedRight, structConf.globalScore);
                
                // ========== UPDATE STABLE ANCHORS ==========
                updateStableAnchors(smoothedLeft, smoothedRight, structConf.globalScore);
                
                // ========== RECOMPUTE WRIST MID ==========
                cv::Point2f smoothedMid = (smoothedLeft + smoothedRight) * 0.5f;
                
                result.wristLeft = smoothedLeft;
                result.wristRight = smoothedRight;
                result.wristMid = smoothedMid;
            }
            
            // ========== PERSIST STATE ==========
            m_lastSmoothedWristLeft = result.wristLeft;
            m_lastSmoothedWristRight = result.wristRight;
            m_lastAngle = currentAngle;
            m_lastAngularVelocity = std::abs(deltaAngle);
            
            if (m_stableFrameCount >= m_stableFramesRequired) {
                m_lastStableAngle = currentAngle;
            }
            
            if (wristConfidence > 0.5f) {
                result.status = "Hand (confident)";
            } else if (wristConfidence > 0.2f) {
                result.status = "Hand (low wrist confidence)";
            } else {
                result.status = "Hand (wrist estimation unreliable)";
            }
            
            // ========== UPDATE GEOMETRYUPDATER ==========
            // GeometryUpdater now receives FINAL AUTHORITATIVE smoothed wrist
            geometryUpdater.updateGeometry(result.palm,
                                          result.wristMid,
                                          result.wristLeft,
                                          result.wristRight,
                                          result.constrainedContour);
            
            if (geometryUpdater.isPalmValid()) {
                result.fingerState = geometryUpdater.getFingerState();
                result.detectedFingerCount = geometryUpdater.getDetectedFingerCount();
                
                std::string stateStr;
                switch (result.fingerState) {
                    case FINGER_STATE_FIST: stateStr = " (fist)"; break;
                    case FINGER_STATE_PARTIAL: stateStr = " (partial)"; break;
                    case FINGER_STATE_OPEN: stateStr = " (open)"; break;
                }
                
                result.status += stateStr;
            }
        }
        
    } catch (const cv::Exception& e) {
        result.status = "Error";
        result.handDetected = false;
    }
    
    return result;
}

bool PalmEstimator::validatePalmShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter) {
    if (contour.size() < 20) return true;
    
    if (!isPointInContour(palmCenter, contour)) {
        return false;
    }
    
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
        return true;
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
        
        contourCenter = projectPointToContourInterior(contourCenter, contour, contourCenter);
        
        float geometryScore = computeGeometryConsistencyScore(contour, contourCenter, 
            std::sqrt(area / M_PI));
        
        return geometryScore > 0.1f;
    }
    
    return true;
}

bool PalmEstimator::fitContourToModel(std::vector<cv::Point>& contour, 
                                     cv::Point2f& palmCenter,
                                     float& handSizeScale) {
    if (contour.empty()) {
        if (lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            handSizeScale = 1.0f;
            return true;
        }
        palmCenter = cv::Point2f(320, 240);
        handSizeScale = 1.0f;
        return true;
    }
    
    std::vector<std::vector<cv::Point>> subContours;
    std::vector<cv::Point> currentSubContour;
    
    for (const auto& p : contour) {
        if (p == cv::Point(-1, -1)) {
            if (!currentSubContour.empty()) {
                subContours.push_back(currentSubContour);
                currentSubContour.clear();
            }
        } else {
            currentSubContour.push_back(p);
        }
    }
    
    if (!currentSubContour.empty()) {
        subContours.push_back(currentSubContour);
    }
    
    if (!subContours.empty()) {
        size_t largestIdx = 0;
        double largestArea = 0;
        for (size_t i = 0; i < subContours.size(); i++) {
            double area = cv::contourArea(subContours[i]);
            if (area > largestArea) {
                largestArea = area;
                largestIdx = i;
            }
        }
        
        contour = subContours[largestIdx];
    }
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) {
        if (lastValidPalm.x >= 0) {
            palmCenter = lastValidPalm;
            handSizeScale = 1.0f;
            return true;
        }
        cv::Rect bbox = cv::boundingRect(contour);
        palmCenter = cv::Point2f(bbox.x + bbox.width/2.0f, bbox.y + bbox.height/2.0f);
        handSizeScale = 1.0f;
    } else {
        palmCenter = cv::Point2f(
            static_cast<float>(m.m10 / m.m00),
            static_cast<float>(m.m01 / m.m00)
        );
        
        palmCenter = projectPointToContourInterior(palmCenter, contour, palmCenter);
    }
    
    if (lastValidPalm.x >= 0) {
        float jumpDist = cv::norm(palmCenter - lastValidPalm);
        if (jumpDist > PE_MAX_PALM_CENTER_JUMP && jumpDist > 0.001f) {
            cv::Point2f direction = (palmCenter - lastValidPalm) * (1.0f / jumpDist);
            palmCenter = lastValidPalm + direction * (PE_MAX_PALM_CENTER_JUMP * 0.5f);
            palmCenter = projectPointToContourInterior(palmCenter, contour, palmCenter);
        }
    }
    
    handSizeScale = 1.0f;
    if (isCalibrated && calibrationData.ratios.handWidth > 0) {
        cv::Rect bbox = cv::boundingRect(contour);
        float currentWidth = static_cast<float>(bbox.width);
        float calibratedWidth = calibrationData.ratios.handWidth;
        
        handSizeScale = currentWidth / calibratedWidth;
        
        if (handSizeScale < 0.25f) handSizeScale = 0.25f;
        if (handSizeScale > 3.5f) handSizeScale = 3.5f;
    }
    
    lastValidPalm = palmCenter;
    failureFrameCount = 0;
    
    return true;
}

std::vector<cv::Point> PalmEstimator::constrainContourWithWrist(const std::vector<cv::Point>& rawContour,
                                                               const cv::Point2f& wristLeft,
                                                               const cv::Point2f& wristRight,
                                                               const cv::Point2f& wristMid,
                                                               float wristConfidence) const {
    if (rawContour.empty() || wristConfidence < 0.1f) {
        return rawContour;
    }
    
    cv::Moments m = cv::moments(rawContour);
    cv::Point2f estimatedPalmCenter;
    if (m.m00 > 0) {
        estimatedPalmCenter = cv::Point2f(static_cast<float>(m.m10 / m.m00),
                                         static_cast<float>(m.m01 / m.m00));
    } else {
        cv::Rect bbox = cv::boundingRect(rawContour);
        estimatedPalmCenter = cv::Point2f(bbox.x + bbox.width/2.0f,
                                         bbox.y + bbox.height/2.0f);
    }
    
    float estimatedPalmRadius = std::sqrt(cv::contourArea(rawContour) / M_PI);
    
    auto skeletalArcs = computeContourDerivedSkeleton(rawContour, estimatedPalmCenter, estimatedPalmRadius);
    
    std::vector<ArcAnalysis> arcAnalyses;
    for (const auto& arc : skeletalArcs) {
        auto [fingerScore, armScore, palmScore] = analyzeSkeletalArc(
            rawContour, arc, estimatedPalmCenter, estimatedPalmRadius);
        
        float arcLength = 0.0f;
        int idx = arc.first;
        while (true) {
            int nextIdx = (idx + 1) % rawContour.size();
            arcLength += cv::norm(cv::Point2f(rawContour[nextIdx]) - 
                                 cv::Point2f(rawContour[idx]));
            if (idx == arc.second) break;
            idx = nextIdx;
        }
        
        ArcAnalysis analysis;
        analysis.arc = arc;
        analysis.fingerScore = fingerScore;
        analysis.armScore = armScore;
        analysis.palmScore = palmScore;
        analysis.arcLength = arcLength;
        arcAnalyses.push_back(analysis);
    }
    
    auto arcSegments = classifyArcsByStructure(rawContour, arcAnalyses, 
                                              estimatedPalmCenter, estimatedPalmRadius);
    
    buildContourSkeleton(rawContour, estimatedPalmCenter, estimatedPalmRadius, arcSegments);
    
    auto pass1Segments = applyWristLineTrimming(rawContour, wristLeft, wristRight, 
                                               wristConfidence, arcSegments);
    
    auto pass2Segments = applyLobeLimiting(rawContour, wristConfidence, 
                                          pass1Segments, estimatedPalmCenter, estimatedPalmRadius);
    
    auto pass3Segments = applyWidthConsistencyCheck(rawContour, wristLeft, wristRight,
                                                   wristConfidence, pass2Segments);
    
    auto finalSegments = applyCalibrationBiasedRefinement(rawContour, wristConfidence,
                                                         pass3Segments);
    
    auto finalContour = flattenArcSegmentsToContour(rawContour, finalSegments);
    
    if (finalContour.empty()) {
        return rawContour;
    }
    
    if (finalContour.size() < rawContour.size() * 0.3f) {
        return rawContour;
    }
    
    return finalContour;
}

std::vector<ArcSegment> PalmEstimator::classifyArcsByStructure(
    const std::vector<cv::Point>& contour,
    const std::vector<ArcAnalysis>& arcAnalyses,
    const cv::Point2f& palmCenter,
    float palmRadius) const {
    
    std::vector<ArcSegment> segments;
    
    if (contour.empty() || arcAnalyses.empty() || palmRadius < 10.0f) {
        return segments;
    }
    
    for (const auto& analysis : arcAnalyses) {
        ArcSegment segment;
        segment.indices = analysis.arc;
        segment.arcLength = analysis.arcLength;
        segment.avgDistance = 0.0f;
        segment.curvature = 0.0f;
        segment.parentArcIdx = -1;
        segment.isProtected = false;
        segment.childArcIndices.clear();
        segment.shortenRatio = 0.0f;
        segment.isSevered = false;
        segment.severBoundary = {-1, -1};
        segment.keptPointIndices.clear();
        
        std::vector<cv::Point2f> arcPoints;
        int idx = segment.indices.first;
        float distanceSum = 0.0f;
        cv::Point2f centroidSum(0, 0);
        int pointCount = 0;
        
        while (true) {
            cv::Point2f p(contour[idx]);
            arcPoints.push_back(p);
            distanceSum += cv::norm(p - palmCenter);
            centroidSum += p;
            pointCount++;
            
            if (idx == segment.indices.second) break;
            idx = (idx + 1) % contour.size();
        }
        
        segment.centroid = centroidSum * (1.0f / pointCount);
        segment.avgDistance = distanceSum / pointCount;
        
        idx = segment.indices.first;
        while (true) {
            segment.keptPointIndices.push_back(idx);
            if (idx == segment.indices.second) break;
            idx = (idx + 1) % contour.size();
        }
        
        if (arcPoints.size() >= 3) {
            float totalAngle = 0.0f;
            for (size_t i = 1; i < arcPoints.size() - 1; i++) {
                cv::Point2f v1 = arcPoints[i] - arcPoints[i-1];
                cv::Point2f v2 = arcPoints[i+1] - arcPoints[i];
                float angle = std::atan2(v2.y, v2.x) - std::atan2(v1.y, v1.x);
                while (angle > M_PI) angle -= 2 * M_PI;
                while (angle < -M_PI) angle += 2 * M_PI;
                totalAngle += std::abs(angle);
            }
            segment.curvature = totalAngle / (arcPoints.size() - 2);
        }
        
        float linearity = 0.0f;
        if (arcPoints.size() >= 2) {
            cv::Point2f startToEnd = arcPoints.back() - arcPoints.front();
            float directLength = cv::norm(startToEnd);
            linearity = (segment.arcLength > 0.001f) ? (directLength / segment.arcLength) : 0.0f;
        }
        
        bool isLong = segment.arcLength > palmRadius * 2.0f;
        bool isStraight = linearity > 0.85f;
        bool isConsistent = analysis.armScore > 0.6f;
        bool isLowCurvature = segment.curvature < 0.15f;
        
        if (isLong && isStraight && isConsistent && isLowCurvature) {
            segment.classification = ARC_ARM;
            segment.confidence = analysis.armScore;
        }
        else if (segment.avgDistance < palmRadius * 1.2f && 
                 analysis.palmScore > 0.5f &&
                 segment.curvature < 0.3f) {
            segment.classification = ARC_PALM;
            segment.confidence = analysis.palmScore;
            segment.isProtected = true;
        }
        else if (segment.avgDistance > palmRadius * 1.2f &&
                analysis.fingerScore > 0.4f &&
                segment.curvature > 0.2f) {
            segment.classification = ARC_FINGER;
            segment.confidence = analysis.fingerScore;
        }
        else {
            segment.classification = ARC_UNKNOWN;
            segment.confidence = 0.3f;
        }
        
        segments.push_back(segment);
    }
    
    return segments;
}

void PalmEstimator::buildContourSkeleton(
    const std::vector<cv::Point>& contour,
    const cv::Point2f& palmCenter,
    float palmRadius,
    std::vector<ArcSegment>& segments) const {
    
    if (segments.empty()) return;
    
    int singlePalmArcIdx = -1;
    float minPalmDistance = std::numeric_limits<float>::max();
    
    for (size_t i = 0; i < segments.size(); i++) {
        if (segments[i].classification == ARC_PALM) {
            float dist = cv::norm(segments[i].centroid - palmCenter);
            if (dist < minPalmDistance) {
                minPalmDistance = dist;
                singlePalmArcIdx = static_cast<int>(i);
            }
        }
    }
    
    for (size_t i = 0; i < segments.size(); i++) {
        if (segments[i].classification == ARC_PALM && static_cast<int>(i) != singlePalmArcIdx) {
            segments[i].classification = ARC_UNKNOWN;
            segments[i].isProtected = false;
            segments[i].confidence = 0.3f;
        }
    }
    
    if (singlePalmArcIdx < 0) {
        float highestPalmScore = 0.0f;
        for (size_t i = 0; i < segments.size(); i++) {
            if (segments[i].avgDistance < palmRadius * 1.5f && segments[i].confidence > highestPalmScore) {
                highestPalmScore = segments[i].confidence;
                singlePalmArcIdx = static_cast<int>(i);
            }
        }
        
        if (singlePalmArcIdx >= 0) {
            segments[singlePalmArcIdx].classification = ARC_PALM;
            segments[singlePalmArcIdx].isProtected = true;
        }
    }
    
    if (singlePalmArcIdx < 0) return;
    
    segments[singlePalmArcIdx].parentArcIdx = -1;
    
    for (size_t i = 0; i < segments.size(); i++) {
        if (i == static_cast<size_t>(singlePalmArcIdx)) continue;
        
        int bestParent = singlePalmArcIdx;
        float minConnectionCost = std::numeric_limits<float>::max();
        
        for (size_t j = 0; j < segments.size(); j++) {
            if (j == i) continue;
            
            int i_end = segments[i].indices.second;
            int j_start = segments[j].indices.first;
            int j_end = segments[j].indices.second;
            
            cv::Point2f i_endPoint(contour[i_end]);
            cv::Point2f j_startPoint(contour[j_start]);
            cv::Point2f j_endPoint(contour[j_end]);
            
            float distToStart = cv::norm(i_endPoint - j_startPoint);
            float distToEnd = cv::norm(i_endPoint - j_endPoint);
            float minDist = std::min(distToStart, distToEnd);
            
            float connectionCost = minDist;
            if (segments[j].classification != segments[i].classification) {
                connectionCost *= 2.0f;
            }
            
            if (connectionCost < minConnectionCost && minDist < palmRadius * 0.5f) {
                minConnectionCost = connectionCost;
                bestParent = static_cast<int>(j);
            }
        }
        
        if (bestParent >= 0 && minConnectionCost < palmRadius * 0.5f) {
            segments[i].parentArcIdx = bestParent;
            segments[bestParent].childArcIndices.push_back(static_cast<int>(i));
        }
    }
    
    for (size_t i = 0; i < segments.size(); i++) {
        if (segments[i].classification == ARC_FINGER || segments[i].classification == ARC_PALM) {
            continue;
        }
        
        int currentIdx = static_cast<int>(i);
        int chainLength = 0;
        float totalLength = 0.0f;
        
        while (currentIdx >= 0) {
            totalLength += segments[currentIdx].arcLength;
            chainLength++;
            
            int parentIdx = segments[currentIdx].parentArcIdx;
            if (parentIdx < 0 || parentIdx == currentIdx) break;
            
            if (segments[parentIdx].childArcIndices.size() > 1) {
                break;
            }
            
            currentIdx = parentIdx;
        }
        
        if (chainLength >= 2 && totalLength > palmRadius * 3.0f) {
            segments[i].classification = ARC_ARM;
            segments[i].confidence = std::min(1.0f, totalLength / (palmRadius * 4.0f));
        }
    }
}

void PalmEstimator::shortenArcSegment(ArcSegment& segment, float shortenRatio, 
                                     const std::vector<cv::Point>& contour) const {
    if (shortenRatio <= 0.0f) return;
    
    int totalPoints = static_cast<int>(segment.keptPointIndices.size());
    int pointsToKeep = static_cast<int>(totalPoints * (1.0f - shortenRatio));
    pointsToKeep = std::max(2, pointsToKeep);
    
    if (pointsToKeep < totalPoints) {
        segment.keptPointIndices.resize(pointsToKeep);
        segment.shortenRatio = shortenRatio;
    }
}

void PalmEstimator::severArcSegment(ArcSegment& segment, const std::vector<cv::Point>& contour) const {
    segment.isSevered = true;
    segment.severBoundary = segment.indices;
    segment.keptPointIndices.clear();
    segment.shortenRatio = 1.0f;
}

std::vector<std::pair<int, int>> PalmEstimator::computeContourDerivedSkeleton(
    const std::vector<cv::Point>& contour,
    const cv::Point2f& palmCenter,
    float palmRadius) const {
    
    std::vector<std::pair<int, int>> skeletalArcs;
    
    if (contour.size() < 20 || palmRadius < 10.0f) {
        return skeletalArcs;
    }
    
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices, false, false);
    
    std::vector<cv::Vec4i> defects;
    if (hullIndices.size() > 3) {
        cv::convexityDefects(contour, hullIndices, defects);
    }
    
    if (!defects.empty()) {
        std::sort(defects.begin(), defects.end(),
            [](const cv::Vec4i& a, const cv::Vec4i& b) { return a[0] < b[0]; });
        
        for (size_t i = 0; i < defects.size(); i++) {
            int startIdx = defects[i][1];
            int endIdx = defects[(i + 1) % defects.size()][0];
            
            if (startIdx != endIdx) {
                int arcLength = (endIdx > startIdx) ? (endIdx - startIdx) : 
                               (static_cast<int>(contour.size()) - startIdx + endIdx);
                
                if (arcLength > 10 && arcLength < static_cast<int>(contour.size()) / 2) {
                    skeletalArcs.emplace_back(startIdx, endIdx);
                }
            }
        }
    } else {
        int segmentSize = static_cast<int>(contour.size()) / 4;
        for (int i = 0; i < 4; i++) {
            int startIdx = i * segmentSize;
            int endIdx = (i + 1) * segmentSize % contour.size();
            skeletalArcs.emplace_back(startIdx, endIdx);
        }
    }
    
    return skeletalArcs;
}

std::tuple<float, float, float> PalmEstimator::analyzeSkeletalArc(
    const std::vector<cv::Point>& contour,
    const std::pair<int, int>& arc,
    const cv::Point2f& palmCenter,
    float palmRadius) const {
    
    float fingerScore = 0.0f;
    float armScore = 0.0f;
    float palmScore = 0.0f;
    
    int startIdx = arc.first;
    int endIdx = arc.second;
    
    std::vector<cv::Point2f> arcPoints;
    int idx = startIdx;
    while (true) {
        arcPoints.emplace_back(contour[idx]);
        if (idx == endIdx) break;
        idx = (idx + 1) % contour.size();
    }
    
    if (arcPoints.size() < 5) {
        return {fingerScore, armScore, palmScore};
    }
    
    cv::Point2f arcCentroid(0, 0);
    for (const auto& p : arcPoints) {
        arcCentroid += p;
    }
    arcCentroid = arcCentroid * (1.0f / arcPoints.size());
    
    std::vector<float> distances;
    float maxDist = 0.0f;
    float minDist = std::numeric_limits<float>::max();
    
    for (const auto& p : arcPoints) {
        float dist = cv::norm(p - palmCenter);
        distances.push_back(dist);
        maxDist = std::max(maxDist, dist);
        minDist = std::min(minDist, dist);
    }
    
    float avgDist = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();
    float distRatio = avgDist / palmRadius;
    
    float curvature = 0.0f;
    if (arcPoints.size() >= 3) {
        for (size_t i = 1; i < arcPoints.size() - 1; i++) {
            cv::Point2f v1 = arcPoints[i] - arcPoints[i-1];
            cv::Point2f v2 = arcPoints[i+1] - arcPoints[i];
            float angle = std::atan2(v2.y, v2.x) - std::atan2(v1.y, v1.x);
            while (angle > M_PI) angle -= 2 * M_PI;
            while (angle < -M_PI) angle += 2 * M_PI;
            curvature += std::abs(angle);
        }
        curvature /= (arcPoints.size() - 2);
    }
    
    if (distRatio > 1.2f && curvature > 0.3f) {
        fingerScore = std::min(1.0f, (distRatio - 1.0f) * curvature * 2.0f);
    }
    
    float arcLength = 0.0f;
    for (size_t i = 1; i < arcPoints.size(); i++) {
        arcLength += cv::norm(arcPoints[i] - arcPoints[i-1]);
    }
    
    cv::Point2f startToEnd = arcPoints.back() - arcPoints.front();
    float directLength = cv::norm(startToEnd);
    float linearity = (arcLength > 0.001f) ? (directLength / arcLength) : 0.0f;
    
    float distVariance = 0.0f;
    for (float d : distances) {
        float diff = d - avgDist;
        distVariance += diff * diff;
    }
    distVariance /= distances.size();
    float widthConsistency = 1.0f - std::min(1.0f, std::sqrt(distVariance) / (avgDist + 0.001f));
    
    if (arcLength > palmRadius * 1.5f && linearity > 0.8f && widthConsistency > 0.7f) {
        armScore = std::min(1.0f, (arcLength / palmRadius) * linearity * widthConsistency / 3.0f);
    }
    
    if (distRatio < 1.0f && curvature < 0.2f) {
        palmScore = std::min(1.0f, (1.0f - distRatio) * (1.0f - curvature * 2.0f));
    }
    
    return {fingerScore, armScore, palmScore};
}

std::vector<ArcSegment> PalmEstimator::applyWristLineTrimming(const std::vector<cv::Point>& contour,
                                                            const cv::Point2f& wristLeft,
                                                            const cv::Point2f& wristRight,
                                                            float wristConfidence,
                                                            std::vector<ArcSegment>& arcSegments) const {
    if (wristLeft.x < 0 || wristRight.x < 0 || contour.empty() || arcSegments.empty()) {
        return arcSegments;
    }
    
    cv::Point2f wristVec = wristRight - wristLeft;
    float wristLen = cv::norm(wristVec);
    if (wristLen < 0.001f) {
        return arcSegments;
    }
    
    cv::Point2f wristDir = wristVec / wristLen;
    cv::Point2f wristNormal(-wristDir.y, wristDir.x);
    
    cv::Point2f contourCentroid(0, 0);
    for (const auto& p : contour) {
        contourCentroid += cv::Point2f(p);
    }
    contourCentroid = contourCentroid * (1.0f / contour.size());
    
    cv::Point2f toCentroid = contourCentroid - wristLeft;
    if (toCentroid.dot(wristNormal) < 0) {
        wristNormal = -wristNormal;
    }
    
    for (auto& segment : arcSegments) {
        if (segment.classification == ARC_PALM) {
            continue;
        }
        else if (segment.classification == ARC_ARM) {
            int armSidePoints = 0;
            int totalPoints = static_cast<int>(segment.keptPointIndices.size());
            
            for (int pointIdx : segment.keptPointIndices) {
                cv::Point2f vecToPoint = cv::Point2f(contour[pointIdx]) - wristLeft;
                float crossProduct = vecToPoint.dot(wristNormal);
                if (crossProduct < 0) {
                    armSidePoints++;
                }
            }
            
            float armRatio = static_cast<float>(armSidePoints) / totalPoints;
            float baseShortenRatio = armRatio * segment.confidence * 0.8f;
            float confidenceScale = 0.3f + wristConfidence * 0.7f;
            float shortenRatio = baseShortenRatio * confidenceScale;
            
            shortenRatio = std::max(0.2f, std::min(0.9f, shortenRatio));
            
            shortenArcSegment(segment, shortenRatio, contour);
        }
    }
    
    return arcSegments;
}

std::vector<ArcSegment> PalmEstimator::applyLobeLimiting(const std::vector<cv::Point>& contour,
                                                       float confidence,
                                                       std::vector<ArcSegment>& arcSegments,
                                                       const cv::Point2f& palmCenter,
                                                       float palmRadius) const {
    if (contour.size() < 30 || arcSegments.empty()) return arcSegments;
    
    std::vector<int> fingerArcIndices;
    for (size_t i = 0; i < arcSegments.size(); i++) {
        if (arcSegments[i].classification == ARC_FINGER && arcSegments[i].confidence > 0.3f) {
            fingerArcIndices.push_back(static_cast<int>(i));
        }
    }
    
    if (fingerArcIndices.size() > 5 && confidence > 0.3f) {
        std::sort(fingerArcIndices.begin(), fingerArcIndices.end(),
            [&arcSegments](int a, int b) {
                return arcSegments[a].confidence < arcSegments[b].confidence;
            });
        
        int arcsToSever = static_cast<int>(fingerArcIndices.size()) - 5;
        for (int i = 0; i < arcsToSever && i < static_cast<int>(fingerArcIndices.size()); i++) {
            int weakArcIdx = fingerArcIndices[i];
            severArcSegment(arcSegments[weakArcIdx], contour);
        }
    }
    
    if (isCalibrated && calibrationData.ratios.maxFingerLength > 0 && palmRadius > 0) {
        float maxAllowedLength = calibrationData.ratios.maxFingerLength * palmRadius * 1.3f;
        
        for (auto& segment : arcSegments) {
            if (segment.classification == ARC_FINGER && !segment.keptPointIndices.empty()) {
                float arcMaxDistance = 0.0f;
                for (int pointIdx : segment.keptPointIndices) {
                    float dist = cv::norm(cv::Point2f(contour[pointIdx]) - palmCenter);
                    arcMaxDistance = std::max(arcMaxDistance, dist);
                }
                
                if (arcMaxDistance > maxAllowedLength) {
                    float excessRatio = arcMaxDistance / maxAllowedLength;
                    float shortenRatio = std::min(0.7f, (excessRatio - 1.0f) * 0.5f);
                    shortenArcSegment(segment, shortenRatio, contour);
                }
            }
        }
    }
    
    return arcSegments;
}

std::vector<ArcSegment> PalmEstimator::applyWidthConsistencyCheck(const std::vector<cv::Point>& contour,
                                                                const cv::Point2f& wristLeft,
                                                                const cv::Point2f& wristRight,
                                                                float confidence,
                                                                std::vector<ArcSegment>& arcSegments) const {
    if (contour.size() < 50 || wristLeft.x < 0 || wristRight.x < 0 || arcSegments.empty()) {
        return arcSegments;
    }
    
    for (auto& segment : arcSegments) {
        if (segment.classification == ARC_ARM && segment.confidence > 0.6f) {
            bool directlyConnectedToPalm = false;
            if (segment.parentArcIdx >= 0 && segment.parentArcIdx < static_cast<int>(arcSegments.size())) {
                const auto& parent = arcSegments[segment.parentArcIdx];
                if (parent.classification == ARC_PALM) {
                    directlyConnectedToPalm = true;
                }
            }
            
            if (!directlyConnectedToPalm) {
                severArcSegment(segment, contour);
                continue;
            }
        }
    }
    
    std::vector<int> armArcIndices;
    for (size_t i = 0; i < arcSegments.size(); i++) {
        if (arcSegments[i].classification == ARC_ARM && arcSegments[i].confidence > 0.4f && 
            !arcSegments[i].keptPointIndices.empty() && !arcSegments[i].isSevered) {
            armArcIndices.push_back(static_cast<int>(i));
        }
    }
    
    for (int armArcIdx : armArcIndices) {
        auto& armArc = arcSegments[armArcIdx];
        
        int chainLength = 1;
        float totalChainLength = armArc.arcLength;
        int currentIdx = armArcIdx;
        
        int parentIdx = armArc.parentArcIdx;
        while (parentIdx >= 0 && parentIdx < static_cast<int>(arcSegments.size())) {
            if (arcSegments[parentIdx].childArcIndices.size() > 1) {
                break;
            }
            chainLength++;
            totalChainLength += arcSegments[parentIdx].arcLength;
            parentIdx = arcSegments[parentIdx].parentArcIdx;
        }
        
        float chainRatio = totalChainLength / (std::sqrt(cv::contourArea(contour) / M_PI) * 3.0f);
        float baseShortenRatio = std::min(0.9f, chainRatio * 0.8f * armArc.confidence);
        float shortenRatio = std::max(0.3f, baseShortenRatio);
        
        shortenArcSegment(armArc, shortenRatio, contour);
    }
    
    return arcSegments;
}

std::vector<ArcSegment> PalmEstimator::applyCalibrationBiasedRefinement(const std::vector<cv::Point>& contour,
                                                                       float confidence,
                                                                       std::vector<ArcSegment>& arcSegments) const {
    if (!isCalibrated || confidence < 0.3f || arcSegments.empty()) {
        return arcSegments;
    }
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) return arcSegments;
    
    cv::Point2f currentPalmCenter(
        static_cast<float>(m.m10 / m.m00),
        static_cast<float>(m.m01 / m.m00)
    );
    
    float currentArea = cv::contourArea(contour);
    float currentPalmRadius = std::sqrt(currentArea / M_PI);
    
    float calibratedPalmRadius = calibrationData.ratios.palmRadius;
    if (calibratedPalmRadius > 0) {
        float sizeRatio = currentPalmRadius / calibratedPalmRadius;
        
        if (sizeRatio > 1.5f) {
            float expectedMax = calibratedPalmRadius * 2.5f;
            
            for (auto& segment : arcSegments) {
                if (segment.classification != ARC_PALM && 
                    segment.avgDistance > expectedMax * 1.5f) {
                    
                    float excessRatio = segment.avgDistance / expectedMax;
                    float shortenRatio = std::min(0.8f, (excessRatio - 1.0f) * 0.4f);
                    
                    shortenArcSegment(segment, shortenRatio, contour);
                }
            }
        }
    }
    
    if (calibrationData.ratios.avgFingerDistance > 0) {
        std::vector<int> highConfidenceFingerIndices;
        for (size_t i = 0; i < arcSegments.size(); i++) {
            if (arcSegments[i].classification == ARC_FINGER && arcSegments[i].confidence > 0.5f) {
                highConfidenceFingerIndices.push_back(static_cast<int>(i));
            }
        }
        
        float expectedFingers = 5.0f;
        if (calibrationData.ratios.avgFingerDistance > 0 && currentPalmRadius > 0) {
            expectedFingers = std::min(5.0f, currentPalmRadius * 2.0f / calibrationData.ratios.avgFingerDistance);
        }
        
        if (highConfidenceFingerIndices.size() > expectedFingers * 1.5f) {
            std::sort(highConfidenceFingerIndices.begin(), highConfidenceFingerIndices.end(),
                     [&arcSegments](int a, int b) {
                         return arcSegments[a].confidence < arcSegments[b].confidence;
                     });
            
            int arcsToSever = static_cast<int>(highConfidenceFingerIndices.size() - expectedFingers);
            arcsToSever = std::max(0, arcsToSever);
            
            for (int i = 0; i < arcsToSever && i < static_cast<int>(highConfidenceFingerIndices.size()); i++) {
                int weakArcIdx = highConfidenceFingerIndices[i];
                severArcSegment(arcSegments[weakArcIdx], contour);
            }
        }
    }
    
    return arcSegments;
}

std::vector<cv::Point> PalmEstimator::flattenArcSegmentsToContour(const std::vector<cv::Point>& originalContour,
                                                                 const std::vector<ArcSegment>& arcSegments) const {
    std::vector<cv::Point> flattened;
    
    std::vector<ArcSegment> sortedSegments = arcSegments;
    std::sort(sortedSegments.begin(), sortedSegments.end(),
        [](const ArcSegment& a, const ArcSegment& b) {
            return a.indices.first < b.indices.first;
        });
    
    bool lastArcWasSevered = false;
    std::vector<cv::Point> openContour;
    
    for (const auto& segment : sortedSegments) {
        if (segment.keptPointIndices.empty()) {
            if (segment.isSevered) {
                lastArcWasSevered = true;
                
                if (!openContour.empty() && !flattened.empty()) {
                    if (openContour.front() != openContour.back()) {
                        openContour.push_back(openContour.front());
                    }
                    
                    flattened.insert(flattened.end(), openContour.begin(), openContour.end());
                    flattened.push_back(cv::Point(-1, -1));
                    
                    openContour.clear();
                }
            }
            continue;
        }
        
        if (lastArcWasSevered && !openContour.empty()) {
            if (openContour.front() != openContour.back()) {
                openContour.push_back(openContour.front());
            }
            
            flattened.insert(flattened.end(), openContour.begin(), openContour.end());
            flattened.push_back(cv::Point(-1, -1));
            
            openContour.clear();
        }
        
        lastArcWasSevered = false;
        
        for (int idx : segment.keptPointIndices) {
            if (idx >= 0 && idx < static_cast<int>(originalContour.size())) {
                openContour.push_back(originalContour[idx]);
            }
        }
    }
    
    if (!openContour.empty()) {
        if (openContour.front() != openContour.back()) {
            openContour.push_back(openContour.front());
        }
        flattened.insert(flattened.end(), openContour.begin(), openContour.end());
    }
    
    if (!flattened.empty() && flattened.back() == cv::Point(-1, -1)) {
        flattened.pop_back();
    }
    
    return flattened;
}

void PalmEstimator::drawDebugStructures(cv::Mat& frame, 
                                       const std::vector<cv::Point>& contour,
                                       const std::vector<ArcSegment>& arcSegments,
                                       const cv::Point2f& palmCenter) const {
    if (contour.empty() || arcSegments.empty()) return;
    
    for (const auto& segment : arcSegments) {
        cv::Scalar color;
        std::string label;
        
        switch (segment.classification) {
            case ARC_PALM:
                color = cv::Scalar(0, 255, 255);
                label = "PALM";
                break;
            case ARC_FINGER:
                color = cv::Scalar(0, 255, 0);
                label = "FINGER";
                break;
            case ARC_ARM:
                color = cv::Scalar(0, 0, 255);
                label = "ARM";
                break;
            case ARC_UNKNOWN:
                color = cv::Scalar(200, 200, 200);
                label = "UNKNOWN";
                break;
        }
        
        cv::circle(frame, segment.centroid, 4, color, -1);
        
        cv::putText(frame, label, 
                   segment.centroid + cv::Point2f(5, 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
        
        std::string confStr = std::to_string((int)(segment.confidence * 100)) + "%";
        cv::putText(frame, confStr,
                   segment.centroid + cv::Point2f(5, 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
        
        if (segment.isSevered) {
            cv::drawMarker(frame, segment.centroid, cv::Scalar(255, 0, 0), 
                          cv::MARKER_TILTED_CROSS, 15, 3);
            cv::putText(frame, "SEVERED",
                       segment.centroid + cv::Point2f(5, 50),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 2);
        }
        
        if (segment.parentArcIdx >= 0 && 
            segment.parentArcIdx < static_cast<int>(arcSegments.size())) {
            const auto& parent = arcSegments[segment.parentArcIdx];
            cv::line(frame, segment.centroid, parent.centroid,
                    cv::Scalar(255, 255, 255), 1);
        }
        
        if (segment.shortenRatio > 0.01f) {
            std::string shortenStr = "S:" + std::to_string((int)(segment.shortenRatio * 100)) + "%";
            cv::putText(frame, shortenStr,
                       segment.centroid + cv::Point2f(5, 35),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    if (palmCenter.x >= 0) {
        cv::circle(frame, palmCenter, 6, cv::Scalar(255, 0, 255), -1);
        cv::putText(frame, "PALM_CENTER", palmCenter + cv::Point2f(10, 0),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 255), 1);
    }
}

void PalmEstimator::drawWristOverlay(cv::Mat& frame, const Result& result) const {
    // Draw raw wrist (cyan, thin)
    if (result.rawWristLeft.x >= 0 && result.rawWristRight.x >= 0) {
        cv::line(frame, result.rawWristLeft, result.rawWristRight, cv::Scalar(255, 255, 0), 1);
        cv::circle(frame, result.rawWristLeft, 3, cv::Scalar(255, 255, 0), -1);
        cv::circle(frame, result.rawWristRight, 3, cv::Scalar(255, 255, 0), -1);
    }
    
    // Draw smoothed wrist (white, thick)
    if (result.wristLeft.x >= 0 && result.wristRight.x >= 0) {
        cv::Scalar wristColor;
        switch (result.freezeState) {
            case FREEZE_STATE_FREE: wristColor = cv::Scalar(0, 255, 0); break;
            case FREEZE_STATE_DAMPED: wristColor = cv::Scalar(0, 255, 255); break;
            case FREEZE_STATE_FROZEN: wristColor = cv::Scalar(255, 0, 255); break;
            default: wristColor = cv::Scalar(255, 255, 255);
        }
        
        cv::line(frame, result.wristLeft, result.wristRight, wristColor, 3);
        cv::circle(frame, result.wristLeft, 5, wristColor, -1);
        cv::circle(frame, result.wristRight, 5, wristColor, -1);
    }
    
    // Draw frozen anchor (magenta dashed)
    if (m_freezeActive && m_frozenWristLeft.x >= 0 && m_frozenWristRight.x >= 0) {
        cv::LineIterator it(frame, m_frozenWristLeft, m_frozenWristRight, 8);
        for (int i = 0; i < it.count; i++, ++it) {
            if (i % 5 < 3) {
                cv::circle(frame, it.pos(), 2, cv::Scalar(255, 0, 255), -1);
            }
        }
        cv::circle(frame, m_frozenWristLeft, 7, cv::Scalar(255, 0, 255), 1);
        cv::circle(frame, m_frozenWristRight, 7, cv::Scalar(255, 0, 255), 1);
    }
}

void PalmEstimator::drawConfidenceOverlay(cv::Mat& frame, const Result& result) const {
    // Global score bar (top right)
    int barWidth = 200;
    int barHeight = 20;
    int barX = frame.cols - barWidth - 10;
    int barY = 90;
    
    cv::rectangle(frame, cv::Rect(barX, barY, barWidth, barHeight), cv::Scalar(50, 50, 50), -1);
    
    int fillWidth = static_cast<int>(barWidth * result.confidence);
    cv::Scalar barColor;
    if (result.confidence > 0.6f) barColor = cv::Scalar(0, 255, 0);
    else if (result.confidence > 0.4f) barColor = cv::Scalar(0, 255, 255);
    else barColor = cv::Scalar(0, 0, 255);
    
    cv::rectangle(frame, cv::Rect(barX, barY, fillWidth, barHeight), barColor, -1);
    cv::rectangle(frame, cv::Rect(barX, barY, barWidth, barHeight), cv::Scalar(255, 255, 255), 1);
    
    std::string confText = "Conf: " + std::to_string((int)(result.confidence * 100)) + "%";
    cv::putText(frame, confText, cv::Point(barX, barY - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // Alpha value
    std::string alphaText = "Alpha: " + std::to_string((int)(result.effectiveAlpha * 100)) + "%";
    cv::putText(frame, alphaText, cv::Point(barX, barY + barHeight + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 255), 1);
    
    // Temporal confidence
    std::string temporalText = "Temp: " + std::to_string((int)(result.temporalConfidence * 100)) + "%";
    cv::putText(frame, temporalText, cv::Point(barX, barY + barHeight + 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 180), 1);
}

void PalmEstimator::drawFreezeOverlay(cv::Mat& frame, const Result& result) const {
    std::string stateText;
    cv::Scalar stateColor;
    
    switch (result.freezeState) {
        case FREEZE_STATE_FREE:
            stateText = "FREE";
            stateColor = cv::Scalar(0, 255, 0);
            break;
        case FREEZE_STATE_DAMPED:
            stateText = "DAMPED";
            stateColor = cv::Scalar(0, 255, 255);
            break;
        case FREEZE_STATE_FROZEN:
            stateText = "FROZEN (" + std::to_string(result.freezeCounter) + "/" + 
                       std::to_string(UNFREEZE_FRAMES_REQUIRED) + ")";
            stateColor = cv::Scalar(255, 0, 255);
            break;
    }
    
    cv::putText(frame, stateText, cv::Point(10, 110),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, stateColor, 2);
}

void PalmEstimator::draw(cv::Mat& frame, const Result& result, 
                        const cv::Mat& skinMask, const cv::Mat& motionMask) {
    static std::vector<cv::Point> lastValidContour;
    static int contourGraceCounter = 0;
    static std::vector<ArcSegment> lastArcSegments;
    static cv::Point2f lastPalmCenter(-1, -1);
    
    if (!result.contour.empty() && result.handDetected) {
        lastValidContour = result.constrainedContour.empty() ? result.contour : result.constrainedContour;
        contourGraceCounter = 10;
        
        if (!lastValidContour.empty()) {
            cv::Moments m = cv::moments(lastValidContour);
            if (m.m00 > 0) {
                lastPalmCenter = cv::Point2f(
                    static_cast<float>(m.m10 / m.m00),
                    static_cast<float>(m.m01 / m.m00)
                );
            }
            
            float estimatedPalmRadius = std::sqrt(cv::contourArea(lastValidContour) / M_PI);
            auto skeletalArcs = computeContourDerivedSkeleton(lastValidContour, lastPalmCenter, estimatedPalmRadius);
            
            std::vector<ArcAnalysis> arcAnalyses;
            for (const auto& arc : skeletalArcs) {
                auto [fingerScore, armScore, palmScore] = analyzeSkeletalArc(
                    lastValidContour, arc, lastPalmCenter, estimatedPalmRadius);
                
                float arcLength = 0.0f;
                int idx = arc.first;
                while (true) {
                    int nextIdx = (idx + 1) % lastValidContour.size();
                    arcLength += cv::norm(cv::Point2f(lastValidContour[nextIdx]) - 
                                         cv::Point2f(lastValidContour[idx]));
                    if (idx == arc.second) break;
                    idx = nextIdx;
                }
                
                ArcAnalysis analysis;
                analysis.arc = arc;
                analysis.fingerScore = fingerScore;
                analysis.armScore = armScore;
                analysis.palmScore = palmScore;
                analysis.arcLength = arcLength;
                arcAnalyses.push_back(analysis);
            }
            
            lastArcSegments = classifyArcsByStructure(lastValidContour, arcAnalyses, 
                                                     lastPalmCenter, estimatedPalmRadius);
            buildContourSkeleton(lastValidContour, lastPalmCenter, estimatedPalmRadius, lastArcSegments);
        }
    } else if (contourGraceCounter > 0) {
        contourGraceCounter--;
    }
    
    if (result.handDetected || contourGraceCounter > 0) {
        if (!result.contour.empty()) {
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{result.contour},
                0, cv::Scalar(100, 100, 100, 128), 1);
        }
        
        if (!result.constrainedContour.empty()) {
            std::vector<std::vector<cv::Point>> subContours;
            std::vector<cv::Point> currentSubContour;
            
            for (const auto& p : result.constrainedContour) {
                if (p == cv::Point(-1, -1)) {
                    if (!currentSubContour.empty()) {
                        subContours.push_back(currentSubContour);
                        currentSubContour.clear();
                    }
                } else {
                    currentSubContour.push_back(p);
                }
            }
            
            if (!currentSubContour.empty()) {
                subContours.push_back(currentSubContour);
            }
            
            for (const auto& subContour : subContours) {
                if (subContour.size() > 2) {
                    cv::drawContours(frame, std::vector<std::vector<cv::Point>>{subContour},
                        0, cv::Scalar(0, 200, 0), 2);
                }
            }
        } else if (!lastValidContour.empty() && contourGraceCounter > 0) {
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{lastValidContour},
                0, cv::Scalar(100, 100, 100), 1);
        }
        
        if (!lastArcSegments.empty() && lastPalmCenter.x >= 0) {
            drawDebugStructures(frame, lastValidContour, lastArcSegments, lastPalmCenter);
        }
        
        // NEW: Full debug overlay
        drawWristOverlay(frame, result);
        drawConfidenceOverlay(frame, result);
        drawFreezeOverlay(frame, result);
        
        if (result.wristMid.x >= 0 && result.palm.x >= 0) {
            cv::line(frame, result.palm, result.wristMid, cv::Scalar(255, 0, 255), 1);
        }
        
        if (result.palm.x >= 0) {
            cv::circle(frame, result.palm, 8, cv::Scalar(0, 255, 255), -1);
            cv::putText(frame, "Palm", result.palm + cv::Point2f(10, 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        }
        
        if (geometryUpdater.isPalmValid()) {
            const auto& handFrame = geometryUpdater.getHandFrame();
            if (handFrame.isValid()) {
                cv::Point2f axisEnd = result.palm + handFrame.primaryAxis * 50.0f;
                cv::Point2f lateralEnd = result.palm + handFrame.secondaryAxis * 50.0f;
                
                cv::arrowedLine(frame, result.palm, axisEnd, cv::Scalar(0, 255, 0), 2);
                cv::arrowedLine(frame, result.palm, lateralEnd, cv::Scalar(255, 0, 0), 2);
                
                cv::putText(frame, "Primary", axisEnd + cv::Point2f(5, 5),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
                cv::putText(frame, "Lateral", lateralEnd + cv::Point2f(5, 5),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
            }
            
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
                    
                    if (result.palm.x >= 0) {
                        cv::line(frame, result.palm, finger.displayTip,
                                 cv::Scalar(200, 200, 200), 1);
                    }
                    
                    std::string status = finger.isLocked ? "L" : std::to_string((int)(finger.confidence * 100));
                    cv::putText(frame, status + "%",
                              finger.displayTip + cv::Point2f(5, 30),
                              cv::FONT_HERSHEY_SIMPLEX, 0.4,
                              finger.isLocked ? cv::Scalar(0, 255, 0) : cv::Scalar(200, 200, 200),
                              1);
                }
            }
            
            std::string stateInfo;
            switch (geometryUpdater.getFingerState()) {
                case FINGER_STATE_FIST: stateInfo = "FIST"; break;
                case FINGER_STATE_PARTIAL: stateInfo = "PARTIAL"; break;
                case FINGER_STATE_OPEN: stateInfo = "OPEN"; break;
            }
            
            stateInfo += " | Fingers: " + std::to_string(geometryUpdater.getDetectedFingerCount());
            if (result.palm.x >= 0) {
                cv::putText(frame, stateInfo, 
                          result.palm + cv::Point2f(-60, -30),
                          cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 200), 1);
            }
        }
    }
    
    cv::rectangle(frame, cv::Rect(0, 0, 640, 80), cv::Scalar(0, 0, 0, 180), -1);
    
    std::string status = result.handDetected ? result.status : result.status;
    cv::Scalar statusColor = result.handDetected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    
    if (!lastArcSegments.empty()) {
        int palmArcs = 0, fingerArcs = 0, armArcs = 0, severedArcs = 0;
        for (const auto& segment : lastArcSegments) {
            if (segment.isSevered) {
                severedArcs++;
                continue;
            }
            if (segment.keptPointIndices.empty()) continue;
            switch (segment.classification) {
                case ARC_PALM: palmArcs++; break;
                case ARC_FINGER: fingerArcs++; break;
                case ARC_ARM: armArcs++; break;
                default: break;
            }
        }
        
        std::string structuralInfo = " | Arcs: P" + std::to_string(palmArcs) +
                                    " F" + std::to_string(fingerArcs) +
                                    " A" + std::to_string(armArcs) +
                                    " SEV" + std::to_string(severedArcs);
        status += structuralInfo;
    }
    
    cv::putText(frame, status, cv::Point(10, 25), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, statusColor, 2);
    
    if (result.wristLeft.x >= 0 && result.wristRight.x >= 0) {
        std::string wristInfo = "Wrist: [" + 
                               std::to_string((int)result.wristLeft.x) + "," + 
                               std::to_string((int)result.wristLeft.y) + "] - [" +
                               std::to_string((int)result.wristRight.x) + "," + 
                               std::to_string((int)result.wristRight.y) + "]";
        wristInfo += " (conf: " + std::to_string((int)(result.confidence * 100)) + "%)";
        cv::putText(frame, wristInfo, cv::Point(10, 50),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 255), 1);
    }
    
    if (isCalibrated && calibrationData.ratios.palmRadius > 0) {
        std::string calInfo = "Calibrated: R=" + 
                             std::to_string((int)calibrationData.ratios.palmRadius) +
                             " MaxF=" + std::to_string((int)calibrationData.ratios.maxFingerLength);
        cv::putText(frame, calInfo, cv::Point(10, 70),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 200, 100), 1);
    }
    
    std::string controls = "C: Calibrate | T: Toggle | ESC: Exit";
    if (isCalibrated) {
        controls += " | R: Reset Calibration";
    }
    cv::putText(frame, controls, cv::Point(10, 480 - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}