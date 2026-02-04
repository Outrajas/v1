#include "FingerAnalyzer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// Constants for FingerAnalyzer (now using FA_ prefix)
constexpr int FINGER_IDENTITY_PERSISTENCE_FRAMES = 8;
constexpr float MIN_FINGER_ANGLE_SEPARATION = 0.6f;
constexpr float MAX_FINGER_REASSIGN_DISTANCE = 60.0f;
constexpr float MIN_FINGER_PALM_DISTANCE_RATIO = 1.2f;
constexpr float MAX_FINGER_ANGULAR_SPREAD = 2.5f;
constexpr float FINGER_DETECTION_DISTANCE_MULTIPLIER_MIN = 1.2f;
constexpr float FINGER_DETECTION_DISTANCE_MULTIPLIER_MAX = 2.0f;
constexpr float FINGER_DETECTION_DEPTH_THRESHOLD_RATIO = 0.12f;
constexpr float FINGER_CANDIDATE_MIN_DISTANCE = 15.0f;
constexpr float FINGER_HEIGHT_RELATIVE_LIMIT = 0.7f;
constexpr int FINGER_MIN_PERSISTENCE_FOR_LOCK = 8;
constexpr float FINGER_CONFIDENCE_LOCK_THRESHOLD = 0.7f;
constexpr float FINGER_REASSIGNMENT_DISTANCE_PENALTY = 2.0f;
constexpr float FINGER_ANGLE_REASSIGN_THRESHOLD = 0.4f;
constexpr int FINGER_LOCK_DECAY_FRAMES = 20;
constexpr float FINGER_PALM_PROJECTION_WEIGHT = 0.6f;
constexpr int HAND_VALIDITY_GRACE_FRAMES = 12;
constexpr float STATE_HYSTERESIS_THRESHOLD = 0.3f;
constexpr float PALM_SMOOTHING_ALPHA = 0.35f;
constexpr float SCALE_SMOOTHING_ALPHA = 0.25f;
constexpr float WRIST_SMOOTHING_RESPONSIVE = 0.3f;
constexpr float WRIST_DIRECTION_SMOOTHING = 0.4f;
constexpr float RELATIVE_VECTOR_SMOOTHING = 0.35f;
constexpr float CONTOUR_PROJECTION_CORRECTION_WEIGHT = 0.3f;
constexpr float MAX_CONTOUR_PROJECTION_DISTANCE = 25.0f;
constexpr float ADAPTIVE_SMOOTHING_VELOCITY_THRESHOLD = 10.0f;
constexpr float MIN_SMOOTHING_ALPHA = 0.15f;
constexpr float MAX_SMOOTHING_ALPHA = 0.7f;
constexpr float WRIST_DISTANCE_WEIGHT_FALLOFF = 0.5f;
constexpr float MAX_WRIST_DELTA_PER_FRAME = 40.0f;
constexpr float WRIST_DISTANCE_RATIO = 1.5f;
constexpr float WRIST_WIDTH_RATIO = 0.8f;
constexpr float WRIST_ANGULAR_TOLERANCE = 0.5f;
constexpr int WRIST_FALLBACK_FRAMES = 15;

// External declarations
extern std::atomic<bool> isCalibrated;  // Changed from bool to std::atomic<bool>
extern CalibrationResult calibrationData;
extern GeometryUpdater geometryUpdater;
extern ShapeAnchoredTracker shapeTracker;

void FingerIdentity::update(const cv::Point2f& newRawTip, float newAngle, float newPalmRelativeAngle, float newLateralProjection) {
    lastTip = rawTip;
    rawTip = newRawTip;
    
    displayTip = newRawTip;
    
    angle = newAngle;
    palmRelativeAngle = newPalmRelativeAngle;
    lateralProjection = newLateralProjection;
    
    if (smoothedAngle == 0.0f) {
        smoothedAngle = newAngle;
    } else {
        float angleDiff = newAngle - smoothedAngle;
        while (angleDiff > M_PI) angleDiff -= 2 * M_PI;
        while (angleDiff < -M_PI) angleDiff += 2 * M_PI;
        smoothedAngle += angleDiff * 0.3f;
    }
    
    isDetected = true;
    persistenceCounter = FINGER_IDENTITY_PERSISTENCE_FRAMES;
    confidence = std::min(1.0f, confidence + 0.25f);
    framesSinceUpdate = 0;
    stableFrames++;
    
    if (confidence >= FINGER_CONFIDENCE_LOCK_THRESHOLD && !isLocked && stableFrames >= FINGER_MIN_PERSISTENCE_FOR_LOCK) {
        isLocked = true;
        lockFrames = 0;
        confidence = 1.0f;
    }
    if (isLocked) {
        lockFrames++;
    }
}

void FingerIdentity::decay() {
    if (persistenceCounter > 0) {
        persistenceCounter--;
    }
    if (persistenceCounter == 0) {
        isDetected = false;
        confidence = std::max(0.0f, confidence - 0.15f);
        stableFrames = std::max(0, stableFrames - 1);
    }
    framesSinceUpdate++;
    
    if (isLocked && framesSinceUpdate > 3) {
        confidence *= 0.85f;
        if (confidence < FINGER_CONFIDENCE_LOCK_THRESHOLD * 0.5f || framesSinceUpdate > FINGER_LOCK_DECAY_FRAMES) {
            isLocked = false;
            lockFrames = 0;
            stableFrames = 0;
        }
    }
    
    if (!isDetected && confidence < 0.2f) {
        displayTip = cv::Point2f(-1, -1);
    }
}

void FingerIdentity::updatePositionOnly(const cv::Point2f& newRawTip, float newPalmRelativeAngle, float newLateralProjection) {
    lastTip = rawTip;
    rawTip = newRawTip;
    
    displayTip = newRawTip;
    
    palmRelativeAngle = newPalmRelativeAngle;
    lateralProjection = newLateralProjection;
    framesSinceUpdate = 0;
    if (isDetected) {
        stableFrames++;
    }
}

float FingerIdentity::getReassignmentPenalty() const {
    if (isLocked) {
        return FINGER_REASSIGNMENT_DISTANCE_PENALTY * (1.0f + confidence);
    }
    return 1.0f + (confidence * 0.8f);
}

bool FingerIdentity::shouldAllowReassignment(float newPalmRelativeAngle) const {
    if (!isLocked) return true;
    
    float angleDiff = std::abs(newPalmRelativeAngle - palmRelativeAngle);
    if (angleDiff > M_PI) angleDiff = 2 * M_PI - angleDiff;
    
    return angleDiff > FINGER_ANGLE_REASSIGN_THRESHOLD;
}

bool ShapeAnchoredTracker::anchorShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter,
                                      const cv::Point2f& thumbBase, const cv::Point2f& pinkyBase,
                                      float palmRadius, float avgFingerDistance, float handScale) {
    if (contour.size() < 50) return false;
    
    anchor.referencePalmCenter = palmCenter;
    anchor.referenceThumbBase = thumbBase;
    anchor.referencePinkyBase = pinkyBase;
    anchor.referencePalmRadius = palmRadius;
    anchor.referenceScale = handScale;
    
    std::vector<cv::Point2f> simplified;
    int step = std::max(1, static_cast<int>(contour.size() / 30));
    for (size_t i = 0; i < contour.size(); i += step) {
        simplified.push_back(cv::Point2f(static_cast<float>(contour[i].x), 
                                        static_cast<float>(contour[i].y)));
    }
    
    if (simplified.size() < 30) return false;
    anchor.referenceContour = simplified;
    anchor.isAnchored = true;
    return true;
}

bool ShapeAnchoredTracker::validateContour(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter, float currentScale) {
    if (!anchor.isAnchored || contour.empty()) return true;
    
    frameCounter++;
    if (frameCounter % 10 != 0) return true;
    
    float scaleRatio = currentScale / anchor.referenceScale;
    return (scaleRatio >= 1.0f - 0.3f && scaleRatio <= 1.0f + 0.3f);
}

void HandGeometryState::reset() {
    rawPalmCenter = cv::Point2f(-1, -1);
    rawThumbBase = cv::Point2f(-1, -1);
    rawPinkyBase = cv::Point2f(-1, -1);
    rawFingerTips.clear();
    rawHandScale = 1.0f;
    rawContour.clear();
    
    palmContour.clear();
    palmCenter = cv::Point2f(-1, -1);
    palmRadius = 0.0f;
    
    smoothedPalmCenter = cv::Point2f(-1, -1);
    smoothedHandScale = 1.0f;
    
    wristLeft = cv::Point2f(-1, -1);
    wristRight = cv::Point2f(-1, -1);
    wristMid = cv::Point2f(-1, -1);
    smoothedWristLeft = cv::Point2f(-1, -1);
    smoothedWristRight = cv::Point2f(-1, -1);
    lastWristLeft = cv::Point2f(-1, -1);
    lastWristRight = cv::Point2f(-1, -1);
    
    thumbBase = cv::Point2f(-1, -1);
    pinkyBase = cv::Point2f(-1, -1);
    
    thumbPinkyBaseWidth = 0.0f;
    wristBaseRatio = 0.0f;
    anatomicalConsistencyCounter = 0;
    anatomicalConfidence = 0.0f;
    
    avgFingerDistance = 0.0f;
    maxFingerSpread = 0.0f;
    handAxis = cv::Point2f(0, 1);
    lateralAxis = cv::Point2f(1, 0);
    
    fingerIdentities.clear();
    lastFrameFingers.clear();
    
    rawMiddleFingerTip = cv::Point2f(-1, -1);
    rawThumbTip = cv::Point2f(-1, -1);
    rawIndexTip = cv::Point2f(-1, -1);
    relMiddleFingerVector = cv::Point2f(-1, -1);
    smoothedRelMiddleFingerVector = cv::Point2f(-1, -1);
    displayMiddleFingerTip = cv::Point2f(-1, -1);
    displayThumbTip = cv::Point2f(-1, -1);
    displayIndexTip = cv::Point2f(-1, -1);
    
    palmValid = false;
    
    lastWristCenter = cv::Point2f(-1, -1);
    lastWristDir = cv::Point2f(0, 1);
    lastLateralDir = cv::Point2f(1, 0);
    lastWristAngle = 0.0f;
    
    fallbackWristLeft = cv::Point2f(-1, -1);
    fallbackWristRight = cv::Point2f(-1, -1);
    wristFallbackCounter = 0;
    
    cachedHull.clear();
    framesSinceHullRecompute = 0;
    frameCounter = 0;
    lastPalmForHull = cv::Point2f(-1, -1);
    lastContourAreaForHull = 0.0f;
    
    lockedMiddleFingerId = -1;
    
    handValidityGraceCounter = 0;
    wasValidLastFrame = false;
    hsvFailureCounter = 0;
    hsvIsRelaxed = false;
    hsvConfidence = 1.0f;
    
    lastRawPalmForContour = cv::Point2f(-1, -1);
    lastSmoothedPalmCenter = cv::Point2f(-1, -1);
    currentSmoothingAlpha = RELATIVE_VECTOR_SMOOTHING;
    
    fingerState = FINGER_STATE_OPEN;
    detectedFingerCount = 0;
}

void HandGeometryState::computeThumbPinkyBaseWidth() {
    thumbPinkyBaseWidth = 0.0f;
    
    if (thumbBase.x >= 0 && pinkyBase.x >= 0) {
        thumbPinkyBaseWidth = cv::norm(thumbBase - pinkyBase);
    } else if (isCalibrated && calibrationData.thumbPinkyBaseWidth > 0) {
        thumbPinkyBaseWidth = calibrationData.thumbPinkyBaseWidth * smoothedHandScale;
    }
}

void HandGeometryState::computeFingerGeometry() {
    avgFingerDistance = 0.0f;
    maxFingerSpread = 0.0f;
    handAxis = cv::Point2f(0, 1);
    
    if (palmCenter.x < 0 || fingerIdentities.empty()) return;
    
    float totalDist = 0.0f;
    cv::Point2f fingerMean(0, 0);
    int validFingers = 0;
    
    for (const auto& finger : fingerIdentities) {
        if (finger.isDetected) {
            float dist = cv::norm(finger.rawTip - palmCenter);
            totalDist += dist;
            fingerMean += finger.rawTip;
            validFingers++;
        }
    }
    
    if (validFingers > 0) {
        avgFingerDistance = totalDist / validFingers;
        fingerMean *= (1.0f / validFingers);
        handAxis = fingerMean - palmCenter;
        float axisNorm = cv::norm(handAxis);
        if (axisNorm > 0.001f) {
            handAxis *= (1.0f / axisNorm);
            lateralAxis = cv::Point2f(-handAxis.y, handAxis.x);
        }
    }
}

cv::Point2f HandGeometryState::projectToContour(const cv::Point2f& point, const std::vector<cv::Point>& contour) {
    if (contour.empty()) return point;
    
    float minDist = std::numeric_limits<float>::max();
    cv::Point2f nearestPoint = point;
    
    for (const auto& contourPoint : contour) {
        cv::Point2f cp(contourPoint);
        float dist = cv::norm(cp - point);
        if (dist < minDist) {
            minDist = dist;
            nearestPoint = cp;
        }
    }
    
    if (minDist > MAX_CONTOUR_PROJECTION_DISTANCE) {
        return point;
    }
    
    return nearestPoint;
}

float HandGeometryState::computeAdaptiveSmoothingAlpha() {
    if (lastSmoothedPalmCenter.x < 0 || smoothedPalmCenter.x < 0) {
        return RELATIVE_VECTOR_SMOOTHING;
    }
    
    cv::Point2f palmDelta = smoothedPalmCenter - lastSmoothedPalmCenter;
    float palmVelocity = cv::norm(palmDelta);
    
    if (palmVelocity < ADAPTIVE_SMOOTHING_VELOCITY_THRESHOLD * 0.5f) {
        return MAX_SMOOTHING_ALPHA;
    } else if (palmVelocity > ADAPTIVE_SMOOTHING_VELOCITY_THRESHOLD * 2.0f) {
        return MIN_SMOOTHING_ALPHA * 0.7f;
    } else {
        float velocityRatio = (palmVelocity - ADAPTIVE_SMOOTHING_VELOCITY_THRESHOLD * 0.5f) / 
                             (ADAPTIVE_SMOOTHING_VELOCITY_THRESHOLD * 1.5f);
        velocityRatio = std::min(1.0f, std::max(0.0f, velocityRatio));
        velocityRatio = velocityRatio * velocityRatio;
        return MAX_SMOOTHING_ALPHA - velocityRatio * (MAX_SMOOTHING_ALPHA - MIN_SMOOTHING_ALPHA);
    }
}

void HandGeometryState::updateFingerTipsFromIdentities() {
    rawThumbTip = cv::Point2f(-1, -1);
    rawIndexTip = cv::Point2f(-1, -1);
    rawMiddleFingerTip = cv::Point2f(-1, -1);
    
    for (const auto& finger : fingerIdentities) {
        if (finger.isDetected) {
            switch (finger.id) {
                case 0: 
                    rawThumbTip = finger.rawTip;
                    displayThumbTip = finger.displayTip;
                    break;
                case 1: 
                    rawIndexTip = finger.rawTip;
                    displayIndexTip = finger.displayTip;
                    break;
                case 2: 
                    rawMiddleFingerTip = finger.rawTip;
                    displayMiddleFingerTip = finger.displayTip;
                    break;
            }
        }
    }
    
    currentSmoothingAlpha = computeAdaptiveSmoothingAlpha();
    
    if (rawMiddleFingerTip.x >= 0 && palmCenter.x >= 0) {
        cv::Point2f newRelVector = rawMiddleFingerTip - palmCenter;
        
        float relLen = cv::norm(newRelVector);
        float maxAllowed = avgFingerDistance * 1.4f;
        
        if (avgFingerDistance > 0 && relLen > maxAllowed) {
            if (smoothedRelMiddleFingerVector.x >= 0) {
                newRelVector = smoothedRelMiddleFingerVector;
            }
        }
        
        if (relMiddleFingerVector.x < 0) {
            relMiddleFingerVector = newRelVector;
            smoothedRelMiddleFingerVector = newRelVector;
        } else {
            relMiddleFingerVector = newRelVector;
            
            smoothedRelMiddleFingerVector = smoothedRelMiddleFingerVector * (1.0f - currentSmoothingAlpha) + 
                                           newRelVector * currentSmoothingAlpha;
            
            if (!palmContour.empty()) {
                cv::Point2f candidatePoint = palmCenter + smoothedRelMiddleFingerVector;
                cv::Point2f projectedPoint = projectToContour(candidatePoint, palmContour);
                
                float projectionDist = cv::norm(projectedPoint - candidatePoint);
                if (projectionDist > 0.1f && projectionDist < MAX_CONTOUR_PROJECTION_DISTANCE) {
                    float weight = CONTOUR_PROJECTION_CORRECTION_WEIGHT * 
                                  std::min(1.0f, projectionDist / 20.0f);
                    
                    cv::Point2f projectionCorrection = projectedPoint - candidatePoint;
                    smoothedRelMiddleFingerVector += projectionCorrection * weight;
                }
            }
        }
        
        if (smoothedPalmCenter.x >= 0) {
            displayMiddleFingerTip = smoothedPalmCenter + smoothedRelMiddleFingerVector;
        } else {
            displayMiddleFingerTip = palmCenter + smoothedRelMiddleFingerVector;
        }
    } else {
        relMiddleFingerVector = cv::Point2f(-1, -1);
        smoothedRelMiddleFingerVector = cv::Point2f(-1, -1);
    }
    
    lastSmoothedPalmCenter = smoothedPalmCenter;
}

FingerState HandGeometryState::classifyFingerState(const std::vector<cv::Point>& contour) {
    if (contour.size() < 50 || palmRadius < 20.0f) {
        return FINGER_STATE_OPEN;
    }
    
    std::vector<int> hullIndices;
    cv::convexHull(contour, hullIndices, false, false);
    
    if (hullIndices.size() < 10) {
        return FINGER_STATE_FIST;
    }
    
    std::vector<cv::Vec4i> defects;
    if (hullIndices.size() > 3) {
        cv::convexityDefects(contour, hullIndices, defects);
    }
    
    int validDefects = 0;
    for (const auto& defect : defects) {
        float depth = defect[3] / 256.0f;
        if (depth > palmRadius * FA_MIN_DEFECT_DEPTH_RATIO) {
            validDefects++;
        }
    }
    
    if (validDefects <= 1) {
        return FINGER_STATE_FIST;
    } else if (validDefects <= 3) {
        return FINGER_STATE_PARTIAL;
    } else {
        return FINGER_STATE_OPEN;
    }
}

void HandGeometryState::matchFingerIdentities(std::vector<FingerIdentity>& currentCandidates) {
    fingerState = classifyFingerState(rawContour);
    detectedFingerCount = static_cast<int>(currentCandidates.size());
    
    if (fingerState == FINGER_STATE_FIST) {
        for (auto& finger : fingerIdentities) {
            finger.isDetected = false;
            finger.persistenceCounter = 0;
            finger.confidence = 0.0f;
            finger.displayTip = cv::Point2f(-1, -1);
        }
        lastFrameFingers = fingerIdentities;
        return;
    }
    
    for (auto& candidate : currentCandidates) {
        cv::Point2f vec = candidate.rawTip - palmCenter;
        candidate.palmRelativeAngle = std::atan2(vec.y, vec.x);
        candidate.lateralProjection = lateralAxis.dot(vec);
        candidate.id = -1;
    }
    
    std::vector<FingerIdentity> matchedFingers;
    std::vector<bool> candidateUsed(currentCandidates.size(), false);
    
    std::sort(currentCandidates.begin(), currentCandidates.end(),
        [](const FingerIdentity& a, const FingerIdentity& b) {
            return a.lateralProjection < b.lateralProjection;
        });
    
    for (auto& lastFinger : lastFrameFingers) {
        if (lastFinger.isDetected || lastFinger.persistenceCounter > 0) {
            int bestMatchIdx = -1;
            float bestMatchScore = std::numeric_limits<float>::max();
            
            for (size_t i = 0; i < currentCandidates.size(); i++) {
                if (candidateUsed[i]) continue;
                if (!lastFinger.shouldAllowReassignment(currentCandidates[i].palmRelativeAngle)) continue;
                
                float dist = cv::norm(currentCandidates[i].rawTip - lastFinger.rawTip);
                float angleDiff = std::abs(currentCandidates[i].palmRelativeAngle - lastFinger.palmRelativeAngle);
                if (angleDiff > M_PI) angleDiff = 2 * M_PI - angleDiff;
                
                float score = dist * lastFinger.getReassignmentPenalty() + angleDiff * 30.0f;
                
                if (score < bestMatchScore) {
                    bestMatchScore = score;
                    bestMatchIdx = static_cast<int>(i);
                }
            }
            
            if (bestMatchIdx >= 0 && bestMatchScore < MAX_FINGER_REASSIGN_DISTANCE * 2.0f) {
                FingerIdentity matched = currentCandidates[bestMatchIdx];
                matched.id = lastFinger.id;
                
                if (lastFinger.isLocked && lastFinger.confidence > 0.5f) {
                    matched.isLocked = true;
                    matched.confidence = std::min(1.0f, lastFinger.confidence * 0.95f);
                    matched.lockFrames = lastFinger.lockFrames + 1;
                    matched.stableFrames = lastFinger.stableFrames + 1;
                }
                
                matched.persistenceCounter = std::min(FINGER_IDENTITY_PERSISTENCE_FRAMES, 
                                                     lastFinger.persistenceCounter + 4);
                matched.smoothedAngle = lastFinger.smoothedAngle * 0.6f + matched.angle * 0.4f;
                matched.confidence = std::min(1.0f, lastFinger.confidence * 0.8f + 0.2f);
                matchedFingers.push_back(matched);
                candidateUsed[bestMatchIdx] = true;
            } else if (lastFinger.persistenceCounter > 0) {
                lastFinger.decay();
                matchedFingers.push_back(lastFinger);
            }
        }
    }
    
    for (size_t i = 0; i < currentCandidates.size(); i++) {
        if (!candidateUsed[i]) {
            bool tooCloseToLocked = false;
            for (const auto& finger : matchedFingers) {
                if (finger.isLocked) {
                    float dist = cv::norm(currentCandidates[i].rawTip - finger.rawTip);
                    if (dist < FINGER_CANDIDATE_MIN_DISTANCE * 0.5f) {
                        tooCloseToLocked = true;
                        break;
                    }
                }
            }
            
            if (!tooCloseToLocked) {
                FingerIdentity newFinger = currentCandidates[i];
                matchedFingers.push_back(newFinger);
            }
        }
    }
    
    if (fingerState == FINGER_STATE_PARTIAL && matchedFingers.size() > 2) {
        std::sort(matchedFingers.begin(), matchedFingers.end(),
            [](const FingerIdentity& a, const FingerIdentity& b) {
                return a.confidence > b.confidence;
            });
        if (matchedFingers.size() > 2) {
            matchedFingers.resize(2);
        }
    }
    
    std::sort(matchedFingers.begin(), matchedFingers.end(),
        [](const FingerIdentity& a, const FingerIdentity& b) {
            return a.lateralProjection < b.lateralProjection;
        });
    
    for (size_t i = 0; i < matchedFingers.size(); i++) {
        if (matchedFingers[i].id < 0 || matchedFingers[i].id > 4) {
            matchedFingers[i].id = static_cast<int>(i);
        }
    }
    
    if (matchedFingers.size() > 5) {
        std::sort(matchedFingers.begin(), matchedFingers.end(),
            [](const FingerIdentity& a, const FingerIdentity& b) {
                return a.confidence > b.confidence;
            });
        matchedFingers.resize(5);
        
        std::sort(matchedFingers.begin(), matchedFingers.end(),
            [](const FingerIdentity& a, const FingerIdentity& b) {
                return a.lateralProjection < b.lateralProjection;
            });
        
        for (size_t i = 0; i < matchedFingers.size(); i++) {
            matchedFingers[i].id = static_cast<int>(i);
        }
    }
    
    fingerIdentities = matchedFingers;
    lastFrameFingers = fingerIdentities;
}

void HandGeometryState::updateAnatomicalConfidence(cv::Point2f& wristCandidate) {
    if (thumbPinkyBaseWidth < 20.0f) {
        anatomicalConfidence = std::max(0.0f, anatomicalConfidence - FA_ANATOMICAL_CONFIDENCE_DECAY);
        return;
    }
    
    float wristDistance = cv::norm(wristCandidate - palmCenter);
    wristBaseRatio = wristDistance / thumbPinkyBaseWidth;
    
    bool ratioValid = (wristBaseRatio >= FA_THUMB_PINKY_BASE_WIDTH_RATIO_MIN && 
                      wristBaseRatio <= FA_THUMB_PINKY_BASE_WIDTH_RATIO_MAX);
    
    if (ratioValid) {
        anatomicalConfidence = std::min(1.0f, anatomicalConfidence + 0.2f);
        anatomicalConsistencyCounter = std::min(anatomicalConsistencyCounter + 1, 3);
    } else {
        anatomicalConfidence = std::max(0.0f, anatomicalConfidence - 0.15f);
        anatomicalConsistencyCounter = std::max(0, anatomicalConsistencyCounter - 1);
    }
}

std::vector<cv::Point> HandGeometryState::weightContourByDistance(const std::vector<cv::Point>& contour) {
    if (contour.empty() || palmCenter.x < 0) return contour;
    
    std::vector<cv::Point> weightedContour;
    
    float referenceDistance = avgFingerDistance * FA_WRIST_FINGER_DISTANCE_RATIO;
    if (isCalibrated && calibrationData.calibratedWristDistance > 0) {
        referenceDistance = calibrationData.calibratedWristDistance * smoothedHandScale;
    }
    
    float maxValidDistance = referenceDistance * FA_MAX_WRIST_DISTANCE_RATIO;
    
    bool useWristClipping = (wristMid.x >= 0 && wristLeft.x >= 0 && wristRight.x >= 0 && 
                             cv::norm(lastWristDir) > 0.001f);
    
    if (useWristClipping) {
        cv::Point2f wristDir = lastWristDir;
        if (cv::norm(wristDir) > 0.001f) {
            wristDir *= (1.0f / cv::norm(wristDir));
        }
        
        cv::Point2f lateralDir(-wristDir.y, wristDir.x);
        
        cv::Point2f wristBase = wristMid - wristDir * (thumbPinkyBaseWidth * 0.3f);
        
        for (const auto& p : contour) {
            cv::Point2f point(p);
            
            float dist = cv::norm(point - palmCenter);
            if (dist > maxValidDistance) continue;
            
            cv::Point2f toPoint = point - wristBase;
            float wristSide = wristDir.dot(toPoint);
            
            if (wristSide < 0) {
                weightedContour.push_back(p);
            }
        }
    } else {
        for (const auto& p : contour) {
            cv::Point2f point(p);
            float dist = cv::norm(point - palmCenter);
            
            if (dist <= maxValidDistance) {
                weightedContour.push_back(p);
            }
        }
    }
    
    return weightedContour.empty() ? contour : weightedContour;
}

std::vector<cv::Point> HandGeometryState::smoothContourIfNeeded(const std::vector<cv::Point>& contour) {
    return contour;
}

void HandGeometryState::inferWrist() {
    if (palmCenter.x < 0 || palmRadius < 1.0f) {
        return;
    }
    
    cv::Point2f wristDir;
    
    if (lastWristDir.x != 0 || lastWristDir.y != 0) {
        wristDir = lastWristDir;
    } else {
        wristDir = cv::Point2f(0, 1);
    }
    
    cv::Point2f targetDir = -handAxis;
    if (cv::norm(targetDir) > 0.001f) {
        targetDir *= (1.0f / cv::norm(targetDir));
        wristDir = wristDir * (1.0f - WRIST_DIRECTION_SMOOTHING) + targetDir * WRIST_DIRECTION_SMOOTHING;
    }
    
    if (cv::norm(wristDir) > 0.001f) {
        wristDir *= (1.0f / cv::norm(wristDir));
    }
    
    float wristDistance;
    if (fingerIdentities.empty() || avgFingerDistance < palmRadius * 1.5f) {
        wristDistance = thumbPinkyBaseWidth * 1.2f;
    } else {
        wristDistance = avgFingerDistance * 1.5f;
    }
    
    if (calibrationData.calibratedWristDistance > 0) {
        wristDistance = calibrationData.calibratedWristDistance * smoothedHandScale;
    }
    
    cv::Point2f wristCenter = palmCenter + wristDir * wristDistance;
    
    updateAnatomicalConfidence(wristCenter);
    
    if (lastWristCenter.x >= 0) {
        cv::Point2f delta = wristCenter - lastWristCenter;
        float deltaLen = cv::norm(delta);
        if (deltaLen > MAX_WRIST_DELTA_PER_FRAME * 2.0f) {
            wristCenter = lastWristCenter + delta * (MAX_WRIST_DELTA_PER_FRAME * 2.0f / deltaLen);
        }
    }
    
    cv::Point2f lateralDir(-wristDir.y, wristDir.x);
    
    float wristHalfWidth = thumbPinkyBaseWidth > 0 ? thumbPinkyBaseWidth * 0.4f : palmRadius * 0.6f;
    
    wristLeft = wristCenter - lateralDir * wristHalfWidth;
    wristRight = wristCenter + lateralDir * wristHalfWidth;
    wristMid = wristCenter;
    
    lastWristLeft = wristLeft;
    lastWristRight = wristRight;
    lastWristCenter = wristCenter;
    lastWristDir = wristDir;
    lastLateralDir = lateralDir;
    
    if (smoothedWristLeft.x < 0) {
        smoothedWristLeft = wristLeft;
        smoothedWristRight = wristRight;
    } else {
        smoothedWristLeft = wristLeft;
        smoothedWristRight = wristRight;
    }
}

void HandGeometryState::isolatePalmContourFast(const std::vector<cv::Point>& contour) {
    if (contour.empty()) {
        palmContour.clear();
        return;
    }
    
    std::vector<cv::Point> weightedContour = weightContourByDistance(contour);
    
    cv::Moments m = cv::moments(weightedContour);
    if (m.m00 == 0) {
        palmContour.clear();
        return;
    }
    
    cv::Point2f newPalmCenter = cv::Point2f(
        static_cast<float>(m.m10 / m.m00),
        static_cast<float>(m.m01 / m.m00)
    );
    
    if (lastWristLeft.x >= 0 && palmCenter.x >= 0) {
        float jumpDist = cv::norm(newPalmCenter - palmCenter);
        if (jumpDist > 100.0f) {
            cv::Point2f direction = (newPalmCenter - palmCenter) * (1.0f / jumpDist);
            newPalmCenter = palmCenter + direction * (100.0f * 0.5f);
        }
    }
    
    palmCenter = newPalmCenter;
    
    std::vector<float> distances;
    for (const auto& p : weightedContour) {
        distances.push_back(cv::norm(cv::Point2f(p) - palmCenter));
    }
    
    if (!distances.empty()) {
        std::sort(distances.begin(), distances.end());
        palmRadius = distances[distances.size() / 2];
    }
    
    palmContour = weightedContour;
    
    computeThumbPinkyBaseWidth();
    computeAnchoredBases();
}

void HandGeometryState::computeAnchoredBases() {
    if (palmCenter.x < 0) {
        thumbBase = cv::Point2f(-1, -1);
        pinkyBase = cv::Point2f(-1, -1);
        return;
    }
    
    if (isCalibrated && calibrationData.isCalibrated) {
        float scale = smoothedHandScale;
        
        if (calibrationData.thumbBaseOffset.x != 0 || calibrationData.thumbBaseOffset.y != 0) {
            thumbBase = palmCenter + calibrationData.thumbBaseOffset * scale;
        }
        
        if (calibrationData.pinkyBaseOffset.x != 0 || calibrationData.pinkyBaseOffset.y != 0) {
            pinkyBase = palmCenter + calibrationData.pinkyBaseOffset * scale;
        }
    }
    
    if (thumbBase.x < 0 || pinkyBase.x < 0) {
        if (!palmContour.empty()) {
            int minX = palmContour[0].x, maxX = palmContour[0].x;
            cv::Point leftPoint = palmContour[0], rightPoint = palmContour[0];
            
            for (const auto& p : palmContour) {
                if (p.x < minX) {
                    minX = p.x;
                    leftPoint = p;
                }
                if (p.x > maxX) {
                    maxX = p.x;
                    rightPoint = p;
                }
            }
            
            thumbBase = cv::Point2f(static_cast<float>(leftPoint.x), 
                                   palmCenter.y + (leftPoint.y - palmCenter.y) * 0.7f);
            pinkyBase = cv::Point2f(static_cast<float>(rightPoint.x), 
                                   palmCenter.y + (rightPoint.y - palmCenter.y) * 0.7f);
        }
    }
}

std::vector<FingerIdentity> HandGeometryState::detectFingerCandidates(const std::vector<cv::Point>& contour) {
    std::vector<FingerIdentity> fingerCandidates;
    if (contour.empty() || palmCenter.x < 0 || palmRadius < 1.0f) {
        return fingerCandidates;
    }
    
    const std::vector<cv::Point>& detectionContour = contour;
    
    std::vector<int> hullIndices;
    cv::convexHull(detectionContour, hullIndices, false, false);
    
    std::vector<cv::Vec4i> defects;
    if (hullIndices.size() > 3) {
        cv::convexityDefects(detectionContour, hullIndices, defects);
    }
    
    struct Candidate {
        cv::Point2f tip;
        float angle;
        float depth;
        float distance;
        float palmRelativeAngle;
        float lateralProjection;
        int hullIndex;
    };
    std::vector<Candidate> candidates;
    
    for (int idx : hullIndices) {
        if (idx < 0 || idx >= (int)detectionContour.size()) continue;
        
        cv::Point2f tip(detectionContour[idx]);
        cv::Point2f vec = tip - palmCenter;
        float dist = cv::norm(vec);
        float angle = std::atan2(vec.y, vec.x);
        float palmRelativeAngle = std::atan2(vec.y, vec.x);
        float lateralProjection = lateralAxis.dot(vec);
        
        float minDist = palmRadius * FINGER_DETECTION_DISTANCE_MULTIPLIER_MIN;
        float maxDist = palmRadius * FINGER_DETECTION_DISTANCE_MULTIPLIER_MAX;
        
        if (dist >= minDist && dist <= maxDist && 
            tip.y < palmCenter.y + dist * FINGER_HEIGHT_RELATIVE_LIMIT) {
            
            float maxDepth = 0.0f;
            for (const auto& defect : defects) {
                int startIdx = defect[0];
                int endIdx = defect[1];
                int farIdx = defect[2];
                float depth = defect[3] / 256.0f;
                
                if (startIdx == idx || endIdx == idx) {
                    if (depth > maxDepth) {
                        maxDepth = depth;
                    }
                }
            }
            
            if (maxDepth > palmRadius * FINGER_DETECTION_DEPTH_THRESHOLD_RATIO) {
                candidates.push_back({tip, angle, maxDepth, dist, palmRelativeAngle, lateralProjection, idx});
            }
        }
    }
    
    if (candidates.empty()) {
        return fingerCandidates;
    }
    
    std::vector<Candidate> filteredCandidates;
    float minDistThreshold = std::max(palmRadius * 0.2f, FINGER_CANDIDATE_MIN_DISTANCE);
    
    for (size_t i = 0; i < candidates.size(); i++) {
        bool keep = true;
        for (size_t j = 0; j < filteredCandidates.size(); j++) {
            float dist = cv::norm(candidates[i].tip - filteredCandidates[j].tip);
            float angleDiff = std::abs(candidates[i].angle - filteredCandidates[j].angle);
            if (angleDiff > M_PI) angleDiff = 2 * M_PI - angleDiff;
            
            if (dist < minDistThreshold && angleDiff < MIN_FINGER_ANGLE_SEPARATION) {
                keep = false;
                float score_i = candidates[i].depth * candidates[i].distance;
                float score_j = filteredCandidates[j].depth * filteredCandidates[j].distance;
                if (score_i > score_j) {
                    filteredCandidates[j] = candidates[i];
                }
                break;
            }
        }
        if (keep && filteredCandidates.size() < 5) {
            filteredCandidates.push_back(candidates[i]);
        }
    }
    
    std::sort(filteredCandidates.begin(), filteredCandidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.lateralProjection < b.lateralProjection;
        });
    
    for (size_t i = 0; i < filteredCandidates.size(); i++) {
        FingerIdentity finger;
        finger.id = -1;
        finger.update(filteredCandidates[i].tip, filteredCandidates[i].angle,
                     filteredCandidates[i].palmRelativeAngle, filteredCandidates[i].lateralProjection);
        finger.confidence = std::min(0.5f, filteredCandidates[i].depth / (palmRadius * 0.3f));
        fingerCandidates.push_back(finger);
    }
    
    return fingerCandidates;
}

void HandGeometryState::updateFingerTracking() {
    if (rawContour.empty()) {
        for (auto& finger : fingerIdentities) {
            finger.decay();
        }
        lastFrameFingers = fingerIdentities;
        fingerState = FINGER_STATE_FIST;
        detectedFingerCount = 0;
        return;
    }
    
    std::vector<FingerIdentity> detectedCandidates = detectFingerCandidates(rawContour);
    
    fingerState = classifyFingerState(rawContour);
    detectedFingerCount = static_cast<int>(detectedCandidates.size());
    
    matchFingerIdentities(detectedCandidates);
}

bool HandGeometryState::shouldRemainValid() const {
    if (palmValid) {
        return true;
    }
    if (handValidityGraceCounter > 0) {
        return true;
    }
    return false;
}

void HandGeometryState::updateValidity(bool newValid) {
    if (newValid) {
        palmValid = true;
        handValidityGraceCounter = HAND_VALIDITY_GRACE_FRAMES;
        wasValidLastFrame = true;
    } else if (handValidityGraceCounter > 0) {
        handValidityGraceCounter--;
        palmValid = true;
    } else if (wasValidLastFrame) {
        static int failureCount = 0;
        failureCount++;
        if (failureCount >= 5) {
            palmValid = false;
            wasValidLastFrame = false;
            failureCount = 0;
        }
    } else {
        palmValid = false;
    }
}

void HandGeometryState::updateHSVConfidence(bool hsvValid, int skinPixels, int dynamicMinSkin) {
    if (hsvValid) {
        hsvFailureCounter = 0;
        hsvIsRelaxed = false;
        hsvConfidence = 1.0f;
    } else if (palmValid) {
        hsvFailureCounter++;
        if (hsvFailureCounter >= 5) {
            hsvIsRelaxed = true;
        }
        
        float skinRatio = static_cast<float>(skinPixels) / dynamicMinSkin;
        hsvConfidence = std::max(0.3f, skinRatio);
    } else {
        hsvFailureCounter = 0;
        hsvIsRelaxed = false;
        hsvConfidence = 1.0f;
    }
}

float HandGeometryState::getOverallConfidence() const {
    return hsvConfidence * anatomicalConfidence;
}

void GeometryUpdater::updateGeometry(const cv::Point2f& rawPalmCenter,
                                   const cv::Point2f& rawThumbBase,
                                   const cv::Point2f& rawPinkyBase,
                                   const std::vector<cv::Point2f>& rawFingerTips,
                                   float rawHandScale,
                                   const std::vector<cv::Point>& contour) {
    
    currentState.frameCounter++;
    
    if (contour.empty()) {
        currentState.updateValidity(false);
        return;
    }
    
    currentState.rawPalmCenter = rawPalmCenter;
    currentState.rawThumbBase = rawThumbBase;
    currentState.rawPinkyBase = rawPinkyBase;
    currentState.rawFingerTips = rawFingerTips;
    currentState.rawHandScale = rawHandScale;
    currentState.rawContour = contour;
    
    if (isCalibrated && shapeTracker.isAnchored()) {
        if (!shapeTracker.validateContour(contour, rawPalmCenter, rawHandScale)) {
            currentState.rawContour = contour;
        }
    }
    
    currentState.isolatePalmContourFast(currentState.rawContour);
    
    if (currentState.palmContour.empty()) {
        currentState.palmContour = contour;
        currentState.palmCenter = rawPalmCenter;
        currentState.palmRadius = 30.0f;
        currentState.updateValidity(true);
    } else {
        currentState.updateValidity(true);
    }
    
    currentState.updateFingerTracking();
    currentState.computeFingerGeometry();
    
    currentState.inferWrist();
    
    currentState.updateFingerTipsFromIdentities();
    
    cv::Point2f currentPalmCenter = currentState.palmCenter;
    if (currentState.smoothedPalmCenter.x < 0) {
        currentState.smoothedPalmCenter = currentPalmCenter;
    } else {
        currentState.smoothedPalmCenter = currentState.smoothedPalmCenter * 0.6f + currentPalmCenter * 0.4f;
    }
    
    if (rawHandScale > 0) {
        if (currentState.smoothedHandScale < 0) {
            currentState.smoothedHandScale = rawHandScale;
        } else {
            currentState.smoothedHandScale = 
                currentState.smoothedHandScale * (1.0f - SCALE_SMOOTHING_ALPHA) + 
                rawHandScale * SCALE_SMOOTHING_ALPHA;
        }
    }
}