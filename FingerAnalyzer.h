#ifndef FINGER_ANALYZER_H
#define FINGER_ANALYZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <atomic>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Prefix all constants with FA_ to avoid conflicts
constexpr float FA_MIN_DEFECT_DEPTH_RATIO = 0.08f;
constexpr float FA_ANATOMICAL_CONFIDENCE_DECAY = 0.1f;
constexpr float FA_THUMB_PINKY_BASE_WIDTH_RATIO_MIN = 0.7f;
constexpr float FA_THUMB_PINKY_BASE_WIDTH_RATIO_MAX = 1.4f;
constexpr float FA_WRIST_FINGER_DISTANCE_RATIO = 1.2f;
constexpr float FA_MAX_WRIST_DISTANCE_RATIO = 1.8f;

// New constants for hand-local coordinate system
constexpr int FA_HAND_REFERENCE_HISTORY = 10;
constexpr float FA_HAND_AXIS_SMOOTHING = 0.15f;
constexpr float FA_LATERAL_AXIS_SMOOTHING = 0.15f;
constexpr float FA_IDENTITY_LOCK_THRESHOLD = 0.85f;
constexpr int FA_IDENTITY_COOLDOWN_FRAMES = 30;
constexpr float FA_FINGER_REASSIGNMENT_COST_THRESHOLD = 200.0f;
constexpr float FA_ROTATION_COMPENSATION_ALPHA = 0.05f;
constexpr float FA_MIN_ROTATION_FOR_REASSIGNMENT = 0.3f; // radians

enum FingerState {
    FINGER_STATE_FIST = 0,
    FINGER_STATE_PARTIAL = 1,
    FINGER_STATE_OPEN = 2
};

// Hand reference frame for rotation-robust tracking
struct HandReferenceFrame {
    cv::Point2f palmCenter;
    cv::Point2f wristDirection;    // Unit vector pointing from palm to wrist
    cv::Point2f lateralDirection;  // Unit vector perpendicular to wristDirection
    float rotationAngle = 0.0f;    // Current hand rotation relative to vertical
    float rotationVelocity = 0.0f; // Rate of rotation change
    
    // History for smoothing
    std::deque<cv::Point2f> wristDirHistory;
    std::deque<cv::Point2f> lateralDirHistory;
    
    void update(const cv::Point2f& newPalm, const cv::Point2f& newWristDir, 
                const cv::Point2f& newLateralDir) {
        palmCenter = newPalm;
        
        // Smooth wrist direction
        if (wristDirHistory.size() >= FA_HAND_REFERENCE_HISTORY) {
            wristDirHistory.pop_front();
        }
        wristDirHistory.push_back(newWristDir);
        
        // Compute smoothed wrist direction
        cv::Point2f smoothedWrist(0, 0);
        for (const auto& dir : wristDirHistory) {
            smoothedWrist += dir;
        }
        if (cv::norm(smoothedWrist) > 0.001f) {
            wristDirection = smoothedWrist * (1.0f / cv::norm(smoothedWrist));
        } else {
            wristDirection = newWristDir;
        }
        
        // Smooth lateral direction
        if (lateralDirHistory.size() >= FA_HAND_REFERENCE_HISTORY) {
            lateralDirHistory.pop_front();
        }
        lateralDirHistory.push_back(newLateralDir);
        
        // Compute smoothed lateral direction
        cv::Point2f smoothedLateral(0, 0);
        for (const auto& dir : lateralDirHistory) {
            smoothedLateral += dir;
        }
        if (cv::norm(smoothedLateral) > 0.001f) {
            lateralDirection = smoothedLateral * (1.0f / cv::norm(smoothedLateral));
        } else {
            lateralDirection = newLateralDir;
        }
        
        // Update rotation angle (angle of wristDirection from vertical)
        rotationAngle = std::atan2(wristDirection.y, wristDirection.x);
    }
    
    // Convert point from camera coordinates to hand-relative coordinates
    cv::Point2f toHandLocal(const cv::Point2f& point) const {
        cv::Point2f relative = point - palmCenter;
        cv::Point2f local;
        local.x = relative.dot(lateralDirection);   // Lateral position
        local.y = relative.dot(wristDirection);     // Along-hand position
        return local;
    }
    
    // Convert hand-relative coordinates back to camera coordinates
    cv::Point2f toCamera(const cv::Point2f& localPoint) const {
        return palmCenter + lateralDirection * localPoint.x + wristDirection * localPoint.y;
    }
    
    // Get angle in hand-relative coordinate system
    float getHandRelativeAngle(const cv::Point2f& point) const {
        cv::Point2f local = toHandLocal(point);
        return std::atan2(local.y, local.x);
    }
    
    void reset() {
        wristDirHistory.clear();
        lateralDirHistory.clear();
        rotationAngle = 0.0f;
        rotationVelocity = 0.0f;
    }
};

struct FingerIdentity {
    int id = -1;
    cv::Point2f rawTip;
    cv::Point2f displayTip = cv::Point2f(-1, -1);
    cv::Point2f lastTip;
    float angle = 0.0f;
    float smoothedAngle = 0.0f;
    bool isDetected = false;
    int persistenceCounter = 0;
    float confidence = 0.0f;
    int framesSinceUpdate = 0;
    bool isLocked = false;
    int lockFrames = 0;
    float palmRelativeAngle = 0.0f;
    float lateralProjection = 0.0f;
    int stableFrames = 0;
    
    // New: Hand-relative tracking
    cv::Point2f handLocalTip;          // Position in hand-local coordinates
    cv::Point2f handLocalVelocity;     // Velocity in hand-local space
    float handRelativeAngle = 0.0f;    // Angle in hand-relative coordinate system
    float normalizedRadius = 0.0f;     // Distance from palm, normalized by hand scale
    
    // New: Temporal coherence metrics
    float angularMomentum = 0.0f;      // Rate of angle change
    int cooldownCounter = 0;           // Frames until reassignment allowed
    cv::Point2f velocityHistory[3];    // Recent velocities for continuity checking
    int velocityHistoryIndex = 0;
    
    void update(const cv::Point2f& newRawTip, float newAngle, float newPalmRelativeAngle, 
                float newLateralProjection, const cv::Point2f& newHandLocalTip,
                float newHandRelativeAngle, float newNormalizedRadius);
    void decay();
    void updatePositionOnly(const cv::Point2f& newRawTip, float newPalmRelativeAngle, 
                           float newLateralProjection, const cv::Point2f& newHandLocalTip,
                           float newHandRelativeAngle, float newNormalizedRadius);
    float getReassignmentPenalty() const;
    bool shouldAllowReassignment(float newHandRelativeAngle, float rotationDelta) const;
    
    // New: Update velocity and momentum
    void updateVelocity(const cv::Point2f& newHandLocalTip);
    float computeContinuityScore(const cv::Point2f& candidateLocalTip, 
                                 float candidateHandRelativeAngle) const;
};

struct FingerCalibration {
    cv::Point2f tip;
    cv::Point2f base;
    cv::Point2f tipVector;
    cv::Point2f baseOffset;
    float length = 0.0f;
    float angle = 0.0f;
    bool isCalibrated = false;
};

struct WristBoundary {
    cv::Point2f left;
    cv::Point2f right;
    cv::Point2f leftOffset;
    cv::Point2f rightOffset;
    float width = 0.0f;
    float verticalOffset = 0.0f;
    bool isCalibrated = false;
};

struct ShapeAnchor {
    std::vector<cv::Point2f> referenceContour;
    cv::Point2f referencePalmCenter;
    cv::Point2f referenceThumbBase;
    cv::Point2f referencePinkyBase;
    float referencePalmRadius;
    float referenceScale;
    bool isAnchored = false;
    
    void reset() {
        referenceContour.clear();
        referencePalmCenter = cv::Point2f(-1, -1);
        referenceThumbBase = cv::Point2f(-1, -1);
        referencePinkyBase = cv::Point2f(-1, -1);
        referencePalmRadius = 0.0f;
        referenceScale = 1.0f;
        isAnchored = false;
    }
};

struct CalibrationResult {
    cv::Point2f palmCenter;
    cv::Point2f thumbBase;
    cv::Point2f pinkyBase;
    std::vector<cv::Point2f> fingerTips;
    std::vector<cv::Point2f> fingerBases;
    
    FingerCalibration fingers[5];
    std::vector<FingerIdentity> fingerIdentities;
    
    WristBoundary wrist;
    
    cv::Point2f thumbBaseOffset;
    cv::Point2f pinkyBaseOffset;
    
    float thumbPinkyBaseWidth = 0.0f;
    float calibratedWristDistance = 0.0f;
    float calibratedWristWidth = 0.0f;
    float wristPalmDistanceRatio = 0.0f;
    
    ShapeAnchor shapeAnchor;
    
    struct DistanceRatios {
        float thumbToWrist = 0.0f;
        float pinkyToWrist = 0.0f;
        float wristWidth = 0.0f;
        float palmRadius = 0.0f;
        float maxFingerLength = 0.0f;
        float handWidth = 0.0f;
        float handHeight = 0.0f;
        float avgFingerDistance = 0.0f;
    } ratios;
    
    cv::Scalar calibratedHSVLower;
    cv::Scalar calibratedHSVUpper;
    cv::Rect hsvSamplingRect;
    
    std::vector<cv::Point> handContour;
    cv::RotatedRect fittedEllipse;
    
    bool isCalibrated = false;
};

class ShapeAnchoredTracker {
private:
    ShapeAnchor anchor;
    float currentAlpha = 0.7f;
    int frameCounter = 0;
    
public:
    ShapeAnchoredTracker() {
        anchor.reset();
    }
    
    void reset() {
        anchor.reset();
        currentAlpha = 0.7f;
        frameCounter = 0;
    }
    
    bool anchorShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter,
                    const cv::Point2f& thumbBase, const cv::Point2f& pinkyBase,
                    float palmRadius, float avgFingerDistance, float handScale);
    
    bool validateContour(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter, float currentScale);
    
    bool isAnchored() const { return anchor.isAnchored; }
};

struct HandGeometryState {
    cv::Point2f rawPalmCenter = cv::Point2f(-1, -1);
    cv::Point2f rawThumbBase = cv::Point2f(-1, -1);
    cv::Point2f rawPinkyBase = cv::Point2f(-1, -1);
    std::vector<cv::Point2f> rawFingerTips;
    float rawHandScale = 1.0f;
    std::vector<cv::Point> rawContour;
    
    std::vector<cv::Point> palmContour;
    cv::Point2f palmCenter = cv::Point2f(-1, -1);
    float palmRadius = 0.0f;
    
    cv::Point2f smoothedPalmCenter = cv::Point2f(-1, -1);
    float smoothedHandScale = 1.0f;
    
    cv::Point2f wristLeft = cv::Point2f(-1, -1);
    cv::Point2f wristRight = cv::Point2f(-1, -1);
    cv::Point2f wristMid = cv::Point2f(-1, -1);
    cv::Point2f smoothedWristLeft = cv::Point2f(-1, -1);
    cv::Point2f smoothedWristRight = cv::Point2f(-1, -1);
    cv::Point2f lastWristLeft = cv::Point2f(-1, -1);
    cv::Point2f lastWristRight = cv::Point2f(-1, -1);
    
    cv::Point2f thumbBase = cv::Point2f(-1, -1);
    cv::Point2f pinkyBase = cv::Point2f(-1, -1);
    
    float thumbPinkyBaseWidth = 0.0f;
    float wristBaseRatio = 0.0f;
    int anatomicalConsistencyCounter = 0;
    float anatomicalConfidence = 0.0f;
    
    float avgFingerDistance = 0.0f;
    float maxFingerSpread = 0.0f;
    cv::Point2f handAxis = cv::Point2f(0, 1);
    cv::Point2f lateralAxis = cv::Point2f(1, 0);
    
    std::vector<FingerIdentity> fingerIdentities;
    std::vector<FingerIdentity> lastFrameFingers;
    
    cv::Point2f rawMiddleFingerTip = cv::Point2f(-1, -1);
    cv::Point2f rawThumbTip = cv::Point2f(-1, -1);
    cv::Point2f rawIndexTip = cv::Point2f(-1, -1);
    
    cv::Point2f relMiddleFingerVector = cv::Point2f(-1, -1);
    cv::Point2f smoothedRelMiddleFingerVector = cv::Point2f(-1, -1);
    
    cv::Point2f displayMiddleFingerTip = cv::Point2f(-1, -1);
    cv::Point2f displayThumbTip = cv::Point2f(-1, -1);
    cv::Point2f displayIndexTip = cv::Point2f(-1, -1);
    
    bool palmValid = false;
    
    cv::Point2f lastWristCenter = cv::Point2f(-1, -1);
    cv::Point2f lastWristDir = cv::Point2f(0, 1);
    cv::Point2f lastLateralDir = cv::Point2f(1, 0);
    float lastWristAngle = 0.0f;
    
    cv::Point2f fallbackWristLeft = cv::Point2f(-1, -1);
    cv::Point2f fallbackWristRight = cv::Point2f(-1, -1);
    int wristFallbackCounter = 0;
    
    std::vector<cv::Point> cachedHull;
    int framesSinceHullRecompute = 0;
    int frameCounter = 0;
    cv::Point2f lastPalmForHull = cv::Point2f(-1, -1);
    float lastContourAreaForHull = 0.0f;
    
    int lockedMiddleFingerId = -1;
    
    int handValidityGraceCounter = 0;
    bool wasValidLastFrame = false;
    
    int hsvFailureCounter = 0;
    bool hsvIsRelaxed = false;
    float hsvConfidence = 1.0f;
    
    cv::Point2f lastRawPalmForContour = cv::Point2f(-1, -1);
    
    cv::Point2f lastSmoothedPalmCenter = cv::Point2f(-1, -1);
    float currentSmoothingAlpha = 0.35f;
    
    FingerState fingerState = FINGER_STATE_OPEN;
    int detectedFingerCount = 0;
    
    // New: Hand reference frame for rotation-robust tracking
    HandReferenceFrame handFrame;
    float handRotation = 0.0f;
    float handRotationVelocity = 0.0f;
    cv::Point2f lastHandAxis;
    cv::Point2f lastLateralAxis;
    
    // New: Multi-layer smoothing buffers
    std::deque<cv::Point2f> palmSmoothingBuffer;
    std::deque<cv::Point2f> wristDirSmoothingBuffer;
    std::deque<cv::Point2f> fingerTipSmoothingBuffer[5];
    
    void reset();
    void computeThumbPinkyBaseWidth();
    void computeFingerGeometry();
    cv::Point2f projectToContour(const cv::Point2f& point, const std::vector<cv::Point>& contour);
    float computeAdaptiveSmoothingAlpha();
    void updateFingerTipsFromIdentities();
    FingerState classifyFingerState(const std::vector<cv::Point>& contour);
    void matchFingerIdentities(std::vector<FingerIdentity>& currentCandidates);
    void updateAnatomicalConfidence(cv::Point2f& wristCandidate);
    std::vector<cv::Point> weightContourByDistance(const std::vector<cv::Point>& contour);
    std::vector<cv::Point> smoothContourIfNeeded(const std::vector<cv::Point>& contour);
    void inferWrist();
    void isolatePalmContourFast(const std::vector<cv::Point>& contour);
    void computeAnchoredBases();
    std::vector<FingerIdentity> detectFingerCandidates(const std::vector<cv::Point>& contour);
    void updateFingerTracking();
    bool shouldRemainValid() const;
    void updateValidity(bool newValid);
    void updateHSVConfidence(bool hsvValid, int skinPixels, int dynamicMinSkin);
    float getOverallConfidence() const;
    
    // New: Hand-local coordinate system methods
    void updateHandReferenceFrame();
    cv::Point2f toHandLocal(const cv::Point2f& point) const;
    cv::Point2f toCamera(const cv::Point2f& localPoint) const;
    float getHandRelativeAngle(const cv::Point2f& point) const;
    
    // New: Multi-layer smoothing methods
    cv::Point2f smoothPalmCenter(const cv::Point2f& newPalm);
    cv::Point2f smoothWristDirection(const cv::Point2f& newWristDir);
    cv::Point2f smoothFingerTip(int fingerId, const cv::Point2f& newTip);
    
    // New: Rotation compensation
    float computeHandRotation() const;
    void compensateRotation(std::vector<FingerIdentity>& candidates);
};

class GeometryUpdater {
private:
    HandGeometryState currentState;
    std::deque<cv::Point2f> palmSmoothingHistory;
    
    int consecutiveAnatomicalFailures = 0;
    const int MAX_ANATOMICAL_FAILURES = 8;
    
public:
    GeometryUpdater() {
        reset();
    }
    
    void reset() {
        currentState.reset();
        palmSmoothingHistory.clear();
        consecutiveAnatomicalFailures = 0;
    }
    
    void updateGeometry(const cv::Point2f& rawPalmCenter,
                       const cv::Point2f& rawThumbBase,
                       const cv::Point2f& rawPinkyBase,
                       const std::vector<cv::Point2f>& rawFingerTips,
                       float rawHandScale,
                       const std::vector<cv::Point>& contour);
    
    const HandGeometryState& getState() const { return currentState; }
    bool isPalmValid() const { return currentState.shouldRemainValid(); }
    float getAnatomicalConfidence() const { return currentState.anatomicalConfidence; }
    cv::Point2f getPalmCenter() const { return currentState.smoothedPalmCenter; }
    float getHandScale() const { return currentState.smoothedHandScale; }
    
    cv::Point2f getWristLeft() const { return currentState.smoothedWristLeft; }
    cv::Point2f getWristRight() const { return currentState.smoothedWristRight; }
    cv::Point2f getWristMid() const { return currentState.wristMid; }
    
    cv::Point2f getThumbBase() const { return currentState.thumbBase; }
    cv::Point2f getPinkyBase() const { return currentState.pinkyBase; }
    
    cv::Point2f getMiddleFingerTip() const { return currentState.displayMiddleFingerTip; }
    cv::Point2f getThumbTip() const { return currentState.displayThumbTip; }
    cv::Point2f getIndexTip() const { return currentState.displayIndexTip; }
    
    float getThumbPinkyBaseWidth() const { return currentState.thumbPinkyBaseWidth; }
    float getWristBaseRatio() const { return currentState.wristBaseRatio; }
    float getMaxFingerSpread() const { return currentState.maxFingerSpread; }
    
    const std::vector<FingerIdentity>& getFingerIdentities() const { return currentState.fingerIdentities; }
    
    FingerState getFingerState() const { return currentState.fingerState; }
    int getDetectedFingerCount() const { return currentState.detectedFingerCount; }
    
    std::vector<cv::Point2f> getWristPoints() const { 
        std::vector<cv::Point2f> points;
        if (currentState.smoothedWristLeft.x >= 0) {
            points.push_back(currentState.smoothedWristLeft);
        }
        if (currentState.smoothedWristRight.x >= 0) {
            points.push_back(currentState.smoothedWristRight);
        }
        return points;
    }
    
    const std::vector<cv::Point>& getPalmContour() const {
        return currentState.palmContour;
    }
    
    bool isHSVRelaxed() const { return currentState.hsvIsRelaxed; }
    float getHSVConfidence() const { return currentState.hsvConfidence; }
    float getOverallConfidence() const { return currentState.getOverallConfidence(); }
    
    void updateHSVConfidence(bool hsvValid, int skinPixels, int dynamicMinSkin) {
        currentState.updateHSVConfidence(hsvValid, skinPixels, dynamicMinSkin);
    }
    
    // New: Get hand reference frame
    const HandReferenceFrame& getHandFrame() const { return currentState.handFrame; }
};

#endif