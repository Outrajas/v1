#include "FingerAnalyzer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// Constants for FingerAnalyzer
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
constexpr int HAND_VALIDITY_GRACE_FRAMES = 12;

// New constants for stable architecture
constexpr float FINGER_VELOCITY_SMOOTHING = 0.3f;
constexpr float ANGULAR_VELOCITY_SMOOTHING = 0.2f;
constexpr float MAX_FINGER_VELOCITY = 50.0f;
constexpr float MAX_ANGULAR_VELOCITY = 0.5f;

// External declarations
extern std::atomic<bool> isCalibrated;
extern CalibrationResult calibrationData;
extern GeometryUpdater geometryUpdater;
extern ShapeAnchoredTracker shapeTracker;

// FingerIdentity methods
void FingerIdentity::update(const cv::Point2f& newRawTip, const cv::Point2f& newHandLocalTip,
                           float newHandLocalAngle, float newHandLocalRadius,
                           const StableHandReferenceFrame& frame) {
    lastTip = rawTip;
    rawTip = newRawTip;
    
    // Store hand-local representation (CRITICAL)
    handLocalTip = newHandLocalTip;
    handLocalAngle = newHandLocalAngle;
    handLocalRadius = newHandLocalRadius;
    
    // Update display tip (projected to contour if needed)
    displayTip = frame.toCamera(newHandLocalTip);
    
    // Update velocity
    updateVelocity(newHandLocalTip, newHandLocalAngle);
    
    // Store history for continuity
    if (localPositionHistory.size() >= 5) {
        localPositionHistory.pop_front();
    }
    localPositionHistory.push_back(newHandLocalTip);
    
    if (localAngleHistory.size() >= 5) {
        localAngleHistory.pop_front();
    }
    localAngleHistory.push_back(newHandLocalAngle);
    
    // Update tracking state
    isDetected = true;
    persistenceCounter = FINGER_IDENTITY_PERSISTENCE_FRAMES;
    confidence = std::min(1.0f, confidence + 0.25f);
    framesSinceUpdate = 0;
    stableFrames++;
    consecutiveUpdates++;
    
    // Hard locking logic
    if (consecutiveUpdates >= FINGER_MIN_PERSISTENCE_FOR_LOCK && 
        confidence >= FA_IDENTITY_LOCK_THRESHOLD && 
        !isLocked) {
        isLocked = true;
        lockFrames = 0;
        confidence = 1.0f;
        cooldownCounter = FA_IDENTITY_COOLDOWN_FRAMES;
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
        consecutiveUpdates = 0;
    }
    
    framesSinceUpdate++;
    
    // Update cooldown
    if (cooldownCounter > 0) {
        cooldownCounter--;
    }
    
    // Lock decay
    if (isLocked && framesSinceUpdate > 3) {
        confidence *= 0.9f;  // Slower decay for locked fingers
        
        // Break lock only on catastrophic failure
        if (framesSinceUpdate > FINGER_LOCK_DECAY_FRAMES * 2) {
            isLocked = false;
            lockFrames = 0;
            stableFrames = 0;
            cooldownCounter = 0;
        }
    }
    
    // Clear display if completely lost
    if (!isDetected && confidence < 0.1f) {
        displayTip = cv::Point2f(-1, -1);
        resetHistory();
    }
}

void FingerIdentity::updatePositionOnly(const cv::Point2f& newRawTip, const cv::Point2f& newHandLocalTip,
                                       float newHandLocalAngle, float newHandLocalRadius) {
    lastTip = rawTip;
    rawTip = newRawTip;
    
    handLocalTip = newHandLocalTip;
    handLocalAngle = newHandLocalAngle;
    handLocalRadius = newHandLocalRadius;
    
    displayTip = newRawTip;  // Will be reprojected by caller
    
    framesSinceUpdate = 0;
    consecutiveUpdates++;
    
    if (isDetected) {
        stableFrames++;
    }
    
    // Reset cooldown when updated
    cooldownCounter = 0;
    
    // Update velocity
    updateVelocity(newHandLocalTip, newHandLocalAngle);
}

void FingerIdentity::updateVelocity(const cv::Point2f& newHandLocalTip, float newHandLocalAngle) {
    // Store velocity history
    velocityHistory[velocityHistoryIndex] = handLocalVelocity;
    velocityHistoryIndex = (velocityHistoryIndex + 1) % 3;
    
    // Compute new velocities if we have history
    if (localPositionHistory.size() >= 2) {
        cv::Point2f newVelocity = newHandLocalTip - localPositionHistory.back();
        
        // Clamp velocity
        float velMag = cv::norm(newVelocity);
        if (velMag > MAX_FINGER_VELOCITY) {
            newVelocity = newVelocity * (MAX_FINGER_VELOCITY / velMag);
        }
        
        handLocalVelocity = handLocalVelocity * (1.0f - FINGER_VELOCITY_SMOOTHING) + 
                           newVelocity * FINGER_VELOCITY_SMOOTHING;
        
        // Angular velocity
        if (localAngleHistory.size() >= 2) {
            float newAngularVel = newHandLocalAngle - localAngleHistory.back();
            while (newAngularVel > M_PI) newAngularVel -= 2 * M_PI;
            while (newAngularVel < -M_PI) newAngularVel += 2 * M_PI;
            
            angularVelocity = angularVelocity * (1.0f - ANGULAR_VELOCITY_SMOOTHING) + 
                             newAngularVel * ANGULAR_VELOCITY_SMOOTHING;
        }
    }
}

float FingerIdentity::computeContinuityScore(const cv::Point2f& candidateLocalTip, 
                                            float candidateHandLocalAngle) const {
    if (!isDetected || handLocalTip.x < 0) {
        return 0.0f;
    }
    
    // 1. Positional continuity (hand-local)
    float posDist = cv::norm(candidateLocalTip - handLocalTip);
    float posScore = std::exp(-posDist / 20.0f);  // 20 pixel scale
    
    // 2. Angular continuity (hand-local)
    float angleDiff = std::abs(candidateHandLocalAngle - handLocalAngle);
    while (angleDiff > M_PI) angleDiff = 2 * M_PI - angleDiff;
    float angleScore = std::exp(-angleDiff / 0.3f);  // 0.3 radian scale
    
    // 3. Velocity continuity (predicted position)
    cv::Point2f predictedPos = handLocalTip + handLocalVelocity;
    float velDist = cv::norm(candidateLocalTip - predictedPos);
    float velScore = std::exp(-velDist / 15.0f);  // 15 pixel scale
    
    // 4. Angular velocity continuity
    float predictedAngle = handLocalAngle + angularVelocity;
    float predictedAngleDiff = std::abs(candidateHandLocalAngle - predictedAngle);
    while (predictedAngleDiff > M_PI) predictedAngleDiff = 2 * M_PI - predictedAngleDiff;
    float angVelScore = std::exp(-predictedAngleDiff / 0.2f);  // 0.2 radian scale
    
    // Weighted combination (emphasize position and angle)
    return posScore * 0.4f + angleScore * 0.3f + velScore * 0.2f + angVelScore * 0.1f;
}

float FingerIdentity::getReassignmentPenalty() const {
    if (isLocked && cooldownCounter > 0) {
        return 10000.0f;  // Effectively infinite during cooldown
    }
    if (isLocked) {
        return 100.0f * (1.0f + confidence);  // Very high for locked fingers
    }
    return 1.0f + (confidence * 0.5f);  // Moderate for unlocked
}

bool FingerIdentity::shouldAllowReassignment(float newHandLocalAngle, float rotationDelta) const {
    if (!isLocked) return true;
    
    // Never allow reassignment during cooldown
    if (cooldownCounter > 0) {
        return false;
    }
    
    // Allow only for:
    // 1. Large rotation of hand
    // 2. Catastrophic velocity discontinuity
    // 3. Complete confidence collapse (handled elsewhere)
    
    float angleDiff = std::abs(newHandLocalAngle - handLocalAngle);
    while (angleDiff > M_PI) angleDiff = 2 * M_PI - angleDiff;
    
    bool velocityDiscontinuity = cv::norm(handLocalVelocity) > FA_FINGER_VELOCITY_DISCONTINUITY_THRESHOLD;
    
    return (rotationDelta > FA_MIN_ROTATION_FOR_REASSIGNMENT) || 
           (angleDiff > 1.0f) ||  // ~57 degrees
           velocityDiscontinuity;
}

void FingerIdentity::resetHistory() {
    localPositionHistory.clear();
    localAngleHistory.clear();
    handLocalVelocity = cv::Point2f(0, 0);
    angularVelocity = 0.0f;
    velocityHistoryIndex = 0;
    for (int i = 0; i < 3; i++) {
        velocityHistory[i] = cv::Point2f(0, 0);
    }
}

// ShapeAnchoredTracker methods
bool ShapeAnchoredTracker::anchorShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter,
                                      const cv::Point2f& wristMid,
                                      float palmRadius, float avgFingerDistance, float handScale) {
    if (contour.size() < 50) return false;
    
    anchor.referencePalmCenter = palmCenter;
    anchor.referenceThumbBase = cv::Point2f(palmCenter.x - 20, palmCenter.y);  // Estimated
    anchor.referencePinkyBase = cv::Point2f(palmCenter.x + 20, palmCenter.y);  // Estimated
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

// HandGeometryState methods
void HandGeometryState::reset() {
    rawPalmCenter = cv::Point2f(-1, -1);
    rawWristMid = cv::Point2f(-1, -1);
    rawContour.clear();
    
    palmCenter = cv::Point2f(-1, -1);
    wristMid = cv::Point2f(-1, -1);
    wristLeft = cv::Point2f(-1, -1);
    wristRight = cv::Point2f(-1, -1);
    palmRadius = 0.0f;
    thumbPinkyBaseWidth = 0.0f;
    
    smoothedPalmCenter = cv::Point2f(-1, -1);
    smoothedWristMid = cv::Point2f(-1, -1);
    smoothedWristLeft = cv::Point2f(-1, -1);
    smoothedWristRight = cv::Point2f(-1, -1);
    
    handFrame.reset();
    
    fingerIdentities.clear();
    lastFrameFingers.clear();
    
    fingerState = FINGER_STATE_OPEN;
    detectedFingerCount = 0;
    palmValid = false;
    handValidityGraceCounter = 0;
    wasValidLastFrame = false;
    
    palmSmoothingBuffer.clear();
    wristSmoothingBuffer.clear();
    wristAngleSmoothingBuffer.clear();
    palmContour.clear();
}

void HandGeometryState::updateRawGeometry(const cv::Point2f& palm, const cv::Point2f& wrist,
                                         const std::vector<cv::Point>& contour) {
    rawPalmCenter = palm;
    rawWristMid = wrist;
    rawContour = contour;
    
    // Simple copy for now (will be smoothed separately)
    palmCenter = palm;
    wristMid = wrist;
    
    // Estimate wrist boundaries (simplified)
    if (palmCenter.x >= 0 && wristMid.x >= 0) {
        cv::Point2f wristDir = wristMid - palmCenter;
        float wristDist = cv::norm(wristDir);
        if (wristDist > 0.001f) {
            wristDir = wristDir * (1.0f / wristDist);
            cv::Point2f lateralDir(-wristDir.y, wristDir.x);
            
            float wristWidth = thumbPinkyBaseWidth > 0 ? thumbPinkyBaseWidth * 0.5f : 40.0f;
            wristLeft = wristMid - lateralDir * wristWidth;
            wristRight = wristMid + lateralDir * wristWidth;
        }
    }
    
    // Store palm contour
    palmContour = contour;
    
    // Compute palm radius (simple median)
    if (!contour.empty()) {
        std::vector<float> distances;
        for (const auto& p : contour) {
            distances.push_back(cv::norm(cv::Point2f(p) - palmCenter));
        }
        std::sort(distances.begin(), distances.end());
        palmRadius = distances[distances.size() / 2];
    }
}

void HandGeometryState::smoothPalmCenter() {
    if (rawPalmCenter.x < 0) return;
    
    // Maintain buffer
    if (palmSmoothingBuffer.size() >= 5) {
        palmSmoothingBuffer.pop_front();
    }
    palmSmoothingBuffer.push_back(rawPalmCenter);
    
    // Weighted average (more recent = higher weight)
    cv::Point2f smoothed(0, 0);
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < palmSmoothingBuffer.size(); i++) {
        float weight = static_cast<float>(i + 1) / palmSmoothingBuffer.size();
        smoothed += palmSmoothingBuffer[i] * weight;
        totalWeight += weight;
    }
    
    if (totalWeight > 0.0f) {
        smoothedPalmCenter = smoothed * (1.0f / totalWeight);
    } else {
        smoothedPalmCenter = rawPalmCenter;
    }
    
    // Apply to current
    palmCenter = smoothedPalmCenter;
}

void HandGeometryState::smoothWristGeometry() {
    if (rawWristMid.x < 0) return;
    
    // Position smoothing
    if (wristSmoothingBuffer.size() >= 5) {
        wristSmoothingBuffer.pop_front();
    }
    wristSmoothingBuffer.push_back(rawWristMid);
    
    cv::Point2f smoothedMid(0, 0);
    float totalWeight = 0.0f;
    
    for (size_t i = 0; i < wristSmoothingBuffer.size(); i++) {
        float weight = static_cast<float>(i + 1) / wristSmoothingBuffer.size();
        smoothedMid += wristSmoothingBuffer[i] * weight;
        totalWeight += weight;
    }
    
    if (totalWeight > 0.0f) {
        smoothedWristMid = smoothedMid * (1.0f / totalWeight);
    } else {
        smoothedWristMid = rawWristMid;
    }
    
    // Angle smoothing (for wrist direction)
    if (rawPalmCenter.x >= 0 && rawWristMid.x >= 0) {
        cv::Point2f rawDir = rawWristMid - rawPalmCenter;
        float rawAngle = std::atan2(rawDir.y, rawDir.x);
        
        if (wristAngleSmoothingBuffer.size() >= 5) {
            wristAngleSmoothingBuffer.pop_front();
        }
        wristAngleSmoothingBuffer.push_back(rawAngle);
        
        // Angle continuity: unwrap before averaging
        std::vector<float> unwrapped;
        float baseAngle = wristAngleSmoothingBuffer[0];
        unwrapped.push_back(baseAngle);
        
        for (size_t i = 1; i < wristAngleSmoothingBuffer.size(); i++) {
            float angle = wristAngleSmoothingBuffer[i];
            float diff = angle - unwrapped[i-1];
            while (diff > M_PI) diff -= 2 * M_PI;
            while (diff < -M_PI) diff += 2 * M_PI;
            unwrapped.push_back(unwrapped[i-1] + diff * FA_HAND_AXIS_SMOOTHING);
        }
        
        // Average unwrapped angles
        float smoothedAngle = 0.0f;
        for (float angle : unwrapped) {
            smoothedAngle += angle;
        }
        smoothedAngle /= unwrapped.size();
        
        // Recompute wrist position from smoothed angle
        float dist = cv::norm(rawDir);
        cv::Point2f smoothedDir(std::cos(smoothedAngle), std::sin(smoothedAngle));
        smoothedWristMid = palmCenter + smoothedDir * dist;
    }
    
    // Update wrist boundaries based on smoothed position
    wristMid = smoothedWristMid;
    
    if (palmCenter.x >= 0 && wristMid.x >= 0) {
        cv::Point2f wristDir = wristMid - palmCenter;
        float wristDist = cv::norm(wristDir);
        if (wristDist > 0.001f) {
            wristDir = wristDir * (1.0f / wristDist);
            cv::Point2f lateralDir(-wristDir.y, wristDir.x);
            
            float wristWidth = thumbPinkyBaseWidth > 0 ? thumbPinkyBaseWidth * 0.5f : 40.0f;
            smoothedWristLeft = wristMid - lateralDir * wristWidth;
            smoothedWristRight = wristMid + lateralDir * wristWidth;
            wristLeft = smoothedWristLeft;
            wristRight = smoothedWristRight;
        }
    }
}

void HandGeometryState::updateHandReferenceFrame() {
    // CRITICAL: Use only palm and wrist geometry, NOT fingers
    if (palmCenter.x >= 0 && wristMid.x >= 0) {
        if (handFrame.isValid()) {
            handFrame.update(palmCenter, wristMid);
        } else {
            handFrame.initialize(palmCenter, wristMid);
        }
    } else {
        // Fallback: maintain previous orientation if available
        cv::Point2f fallbackDir = handFrame.isValid() ? handFrame.primaryAxis : cv::Point2f(0, 1);
        handFrame.fallbackUpdate(palmCenter, fallbackDir);
    }
}

std::vector<FingerIdentity> HandGeometryState::detectFingerCandidates() {
    std::vector<FingerIdentity> candidates;
    
    if (rawContour.empty() || palmCenter.x < 0 || palmRadius < 10.0f) {
        return candidates;
    }
    
    // Find convex hull
    std::vector<int> hullIndices;
    cv::convexHull(rawContour, hullIndices, false, false);
    
    // Find convexity defects
    std::vector<cv::Vec4i> defects;
    if (hullIndices.size() > 3) {
        cv::convexityDefects(rawContour, hullIndices, defects);
    }
    
    // Analyze hull points as potential finger tips
    for (int idx : hullIndices) {
        if (idx < 0 || idx >= (int)rawContour.size()) continue;
        
        cv::Point2f tip(rawContour[idx]);
        cv::Point2f vec = tip - palmCenter;
        float dist = cv::norm(vec);
        
        // Basic finger criteria
        float minDist = palmRadius * FINGER_DETECTION_DISTANCE_MULTIPLIER_MIN;
        float maxDist = palmRadius * FINGER_DETECTION_DISTANCE_MULTIPLIER_MAX;
        
        if (dist >= minDist && dist <= maxDist) {
            // Check if it's likely a finger (has deep convexity defect)
            bool hasDeepDefect = false;
            float maxDepth = 0.0f;
            
            for (const auto& defect : defects) {
                int startIdx = defect[0];
                int endIdx = defect[1];
                float depth = defect[3] / 256.0f;
                
                if ((startIdx == idx || endIdx == idx) && depth > maxDepth) {
                    maxDepth = depth;
                    if (depth > palmRadius * FINGER_DETECTION_DEPTH_THRESHOLD_RATIO) {
                        hasDeepDefect = true;
                    }
                }
            }
            
            if (hasDeepDefect || maxDepth > 0) {
                // Convert to hand-local coordinates
                cv::Point2f localTip = handFrame.toHandLocal(tip);
                float localAngle = std::atan2(localTip.y, localTip.x);
                float localRadius = cv::norm(localTip) / palmRadius;
                
                // Create candidate
                FingerIdentity candidate;
                candidate.rawTip = tip;
                candidate.handLocalTip = localTip;
                candidate.handLocalAngle = localAngle;
                candidate.handLocalRadius = localRadius;
                candidate.confidence = std::min(1.0f, maxDepth / (palmRadius * 0.3f));
                
                candidates.push_back(candidate);
            }
        }
    }
    
    // Filter nearby candidates (in hand-local space)
    std::vector<FingerIdentity> filtered;
    float minDistThreshold = std::max(palmRadius * 0.15f, FINGER_CANDIDATE_MIN_DISTANCE);
    
    for (auto& candidate : candidates) {
        bool keep = true;
        
        for (auto& existing : filtered) {
            float dist = cv::norm(candidate.handLocalTip - existing.handLocalTip);
            float angleDiff = std::abs(candidate.handLocalAngle - existing.handLocalAngle);
            while (angleDiff > M_PI) angleDiff = 2 * M_PI - angleDiff;
            
            if (dist < minDistThreshold && angleDiff < MIN_FINGER_ANGLE_SEPARATION) {
                keep = false;
                // Keep the better candidate
                if (candidate.confidence > existing.confidence) {
                    existing = candidate;
                }
                break;
            }
        }
        
        if (keep && filtered.size() < 5) {
            filtered.push_back(candidate);
        }
    }
    
    // Sort by lateral position (hand-local x)
    std::sort(filtered.begin(), filtered.end(),
        [](const FingerIdentity& a, const FingerIdentity& b) {
            return a.handLocalTip.x < b.handLocalTip.x;
        });
    
    return filtered;
}

void HandGeometryState::matchFingerIdentities(std::vector<FingerIdentity>& candidates) {
    detectedFingerCount = static_cast<int>(candidates.size());
    
    if (fingerState == FINGER_STATE_FIST) {
        // Clear all fingers in fist state
        for (auto& finger : fingerIdentities) {
            finger.isDetected = false;
            finger.persistenceCounter = 0;
            finger.confidence = 0.0f;
            finger.displayTip = cv::Point2f(-1, -1);
        }
        lastFrameFingers = fingerIdentities;
        return;
    }
    
    // Get hand rotation for reassignment decisions
    float rotationDelta = std::abs(handFrame.primaryAngleVelocity);
    
    // Clear used flags
    std::vector<bool> candidateUsed(candidates.size(), false);
    std::vector<FingerIdentity> matchedFingers;
    
    // Phase 1: Match existing fingers with high continuity
    for (auto& lastFinger : lastFrameFingers) {
        if (lastFinger.isDetected || lastFinger.persistenceCounter > 0) {
            int bestMatchIdx = -1;
            float bestContinuity = -1.0f;
            
            for (size_t i = 0; i < candidates.size(); i++) {
                if (candidateUsed[i]) continue;
                
                // Check if reassignment is allowed for this finger
                if (!lastFinger.shouldAllowReassignment(candidates[i].handLocalAngle, rotationDelta)) {
                    continue;
                }
                
                // Compute continuity score
                float continuity = lastFinger.computeContinuityScore(
                    candidates[i].handLocalTip,
                    candidates[i].handLocalAngle
                );
                
                if (continuity > bestContinuity) {
                    bestContinuity = continuity;
                    bestMatchIdx = static_cast<int>(i);
                }
            }
            
            if (bestMatchIdx >= 0 && bestContinuity > 0.5f) {
                // Strong match - update existing finger
                FingerIdentity matched = candidates[bestMatchIdx];
                matched.id = lastFinger.id;
                
                // Preserve lock state
                if (lastFinger.isLocked) {
                    matched.isLocked = true;
                    matched.lockFrames = lastFinger.lockFrames + 1;
                    matched.stableFrames = lastFinger.stableFrames + 1;
                    matched.cooldownCounter = lastFinger.cooldownCounter > 0 ? 
                                             lastFinger.cooldownCounter - 1 : 0;
                    
                    // Preserve history for continuity
                    matched.localPositionHistory = lastFinger.localPositionHistory;
                    matched.localAngleHistory = lastFinger.localAngleHistory;
                    matched.handLocalVelocity = lastFinger.handLocalVelocity;
                    matched.angularVelocity = lastFinger.angularVelocity;
                }
                
                // Update with new position
                matched.update(matched.rawTip, matched.handLocalTip, 
                             matched.handLocalAngle, matched.handLocalRadius,
                             handFrame);
                
                matchedFingers.push_back(matched);
                candidateUsed[bestMatchIdx] = true;
            } else if (lastFinger.persistenceCounter > 0) {
                // Keep decaying finger
                lastFinger.decay();
                matchedFingers.push_back(lastFinger);
            }
        }
    }
    
    // Phase 2: Assign new IDs to unmatched candidates
    for (size_t i = 0; i < candidates.size(); i++) {
        if (!candidateUsed[i]) {
            FingerIdentity newFinger = candidates[i];
            
            // Check if too close to any locked finger
            bool tooClose = false;
            for (const auto& finger : matchedFingers) {
                if (finger.isLocked && finger.cooldownCounter > 0) {
                    float dist = cv::norm(newFinger.handLocalTip - finger.handLocalTip);
                    if (dist < FINGER_CANDIDATE_MIN_DISTANCE) {
                        tooClose = true;
                        break;
                    }
                }
            }
            
            if (!tooClose) {
                // Initialize new finger
                newFinger.update(newFinger.rawTip, newFinger.handLocalTip,
                               newFinger.handLocalAngle, newFinger.handLocalRadius,
                               handFrame);
                matchedFingers.push_back(newFinger);
            }
        }
    }
    
    // Phase 3: Apply finger state constraints
    if (fingerState == FINGER_STATE_PARTIAL && matchedFingers.size() > 2) {
        // Keep only highest confidence fingers
        std::sort(matchedFingers.begin(), matchedFingers.end(),
            [](const FingerIdentity& a, const FingerIdentity& b) {
                return a.confidence > b.confidence;
            });
        if (matchedFingers.size() > 2) {
            matchedFingers.resize(2);
        }
    }
    
    // Phase 4: Sort by lateral position and assign IDs
    std::sort(matchedFingers.begin(), matchedFingers.end(),
        [](const FingerIdentity& a, const FingerIdentity& b) {
            return a.handLocalTip.x < b.handLocalTip.x;
        });
    
    // Assign consistent IDs
    for (size_t i = 0; i < matchedFingers.size(); i++) {
        if (matchedFingers[i].id < 0) {
            matchedFingers[i].id = static_cast<int>(i);
        }
    }
    
    // Limit to 5 fingers
    if (matchedFingers.size() > 5) {
        matchedFingers.resize(5);
    }
    
    // Update state
    fingerIdentities = matchedFingers;
    lastFrameFingers = fingerIdentities;
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
    
    // Detect candidates
    std::vector<FingerIdentity> candidates = detectFingerCandidates();
    
    // Update finger state
    fingerState = classifyFingerState();
    detectedFingerCount = static_cast<int>(candidates.size());
    
    // Match identities
    matchFingerIdentities(candidates);
}

cv::Point2f HandGeometryState::projectToContour(const cv::Point2f& point) const {
    if (palmContour.empty()) return point;
    
    float minDist = std::numeric_limits<float>::max();
    cv::Point2f nearest = point;
    
    for (const auto& p : palmContour) {
        cv::Point2f contourPoint(p);
        float dist = cv::norm(contourPoint - point);
        if (dist < minDist) {
            minDist = dist;
            nearest = contourPoint;
        }
    }
    
    return nearest;
}

FingerState HandGeometryState::classifyFingerState() const {
    if (palmContour.size() < 50 || palmRadius < 20.0f) {
        return FINGER_STATE_OPEN;
    }
    
    // Simple classification based on detected fingers
    int validFingers = 0;
    for (const auto& finger : fingerIdentities) {
        if (finger.isDetected && finger.confidence > 0.3f) {
            validFingers++;
        }
    }
    
    if (validFingers == 0) {
        return FINGER_STATE_FIST;
    } else if (validFingers <= 2) {
        return FINGER_STATE_PARTIAL;
    } else {
        return FINGER_STATE_OPEN;
    }
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

// GeometryUpdater methods
void GeometryUpdater::updateGeometry(const cv::Point2f& rawPalmCenter,
                                   const cv::Point2f& rawWristMid,
                                   const cv::Point2f& rawWristLeft,
                                   const cv::Point2f& rawWristRight,
                                   const std::vector<cv::Point>& contour) {
    
    // Store raw inputs
    currentState.updateRawGeometry(rawPalmCenter, rawWristMid, contour);
    
    // Independent smoothing paths (NO CROSS-DEPENDENCIES)
    currentState.smoothPalmCenter();
    currentState.smoothWristGeometry();
    
    // Update hand reference frame (based on smoothed geometry)
    currentState.updateHandReferenceFrame();
    
    // Update finger tracking (in hand-local space only)
    currentState.updateFingerTracking();
    
    // Update validity
    currentState.updateValidity(!contour.empty());
}