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

// New constants for stable architecture
constexpr int FA_HAND_REFERENCE_HISTORY = 10;
constexpr float FA_HAND_AXIS_SMOOTHING = 0.1f;  // Lower for stability
constexpr float FA_IDENTITY_LOCK_THRESHOLD = 0.85f;
constexpr int FA_IDENTITY_COOLDOWN_FRAMES = 60;  // Longer cooldown
constexpr float FA_FINGER_REASSIGNMENT_COST_THRESHOLD = 300.0f;  // Higher threshold
constexpr float FA_MIN_ROTATION_FOR_REASSIGNMENT = 0.5f;  // radians, larger threshold
constexpr float FA_FINGER_VELOCITY_DISCONTINUITY_THRESHOLD = 30.0f;  // pixels/frame

// Wrist model constants
constexpr float FA_WRIST_DISTANCE_RATIO_MIN = 0.5f;   // min wrist distance / thumb-pinky width
constexpr float FA_WRIST_DISTANCE_RATIO_MAX = 1.2f;   // max wrist distance / thumb-pinky width
constexpr float FA_WRIST_WIDTH_RATIO = 0.6f;          // wrist width / thumb-pinky width
constexpr float FA_MIN_THUMB_PINKY_WIDTH = 30.0f;     // minimum width to avoid division by zero
constexpr float FA_BASE_MIN_Y_RELATIVE = 0.1f;        // relative to palm radius for base extraction
constexpr float FA_WRIST_MOVEMENT_CORROBORATION_ANGLE_THRESHOLD = 0.3f; // radians
constexpr float FA_TEMPORAL_DECAY_RATE = 0.9f;        // decay rate for lost bases
constexpr int FA_MIN_CONTOUR_POINTS = 20;             // minimum points for base extraction

// Hand orientation modes
enum HandOrientationMode {
    HAND_ORIENTATION_WRIST_BASED,     // Primary: wrist direction
    HAND_ORIENTATION_FALLBACK_AXIS    // Fallback: hand axis (only when wrist unavailable)
};

// Stable hand reference frame - CRITICAL FIX
struct StableHandReferenceFrame {
    cv::Point2f palmCenter;
    cv::Point2f wristMid;            // Absolute wrist position
    cv::Point2f primaryAxis;         // Unit vector: palm -> wrist (stable reference)
    cv::Point2f secondaryAxis;       // Unit vector: perpendicular to primary
    float primaryAngle = 0.0f;       // Angle of primary axis (stable, unwrapped)
    float primaryAngleVelocity = 0.0f; // Rate of angle change
    HandOrientationMode mode = HAND_ORIENTATION_WRIST_BASED;
    
    // History for angular continuity (prevents π flips)
    std::deque<float> angleHistory;
    float lastStableAngle = 0.0f;
    
    // Initialize with palm and wrist
    void initialize(const cv::Point2f& palm, const cv::Point2f& wrist) {
        palmCenter = palm;
        wristMid = wrist;
        
        // Primary axis: palm -> wrist (NOT from fingers!)
        primaryAxis = wrist - palm;
        float norm = cv::norm(primaryAxis);
        if (norm > 0.001f) {
            primaryAxis = primaryAxis * (1.0f / norm);
        } else {
            primaryAxis = cv::Point2f(0, 1);  // Default down
        }
        
        // Secondary axis: perpendicular (consistent orientation)
        secondaryAxis = cv::Point2f(-primaryAxis.y, primaryAxis.x);
        
        // Initial angle
        primaryAngle = std::atan2(primaryAxis.y, primaryAxis.x);
        lastStableAngle = primaryAngle;
        angleHistory.clear();
        angleHistory.push_back(primaryAngle);
        
        mode = HAND_ORIENTATION_WRIST_BASED;
    }
    
    // Update with angle continuity (NO π flips)
    void update(const cv::Point2f& newPalm, const cv::Point2f& newWrist) {
        palmCenter = newPalm;
        wristMid = newWrist;
        
        // Compute new primary axis
        cv::Point2f newPrimary = wristMid - palmCenter;
        float norm = cv::norm(newPrimary);
        if (norm > 0.001f) {
            newPrimary = newPrimary * (1.0f / norm);
        } else {
            // Fallback: maintain previous direction
            newPrimary = primaryAxis;
        }
        
        // Compute new angle with continuity
        float newAngle = std::atan2(newPrimary.y, newPrimary.x);
        
        // Angle unwrapping to prevent π flips
        float angleDiff = newAngle - lastStableAngle;
        while (angleDiff > M_PI) angleDiff -= 2 * M_PI;
        while (angleDiff < -M_PI) angleDiff += 2 * M_PI;
        
        // Apply angular smoothing
        primaryAngle = lastStableAngle + angleDiff * FA_HAND_AXIS_SMOOTHING;
        
        // Update velocity
        primaryAngleVelocity = angleDiff;
        
        // Recompute axis from smoothed angle
        primaryAxis = cv::Point2f(std::cos(primaryAngle), std::sin(primaryAngle));
        secondaryAxis = cv::Point2f(-primaryAxis.y, primaryAxis.x);
        
        // Update history
        if (angleHistory.size() >= FA_HAND_REFERENCE_HISTORY) {
            angleHistory.pop_front();
        }
        angleHistory.push_back(primaryAngle);
        lastStableAngle = primaryAngle;
    }
    
    // Fallback: use when wrist is not available
    void fallbackUpdate(const cv::Point2f& newPalm, const cv::Point2f& fallbackDirection) {
        palmCenter = newPalm;
        
        // Use provided fallback direction (e.g., from previous frame)
        primaryAxis = fallbackDirection;
        float norm = cv::norm(primaryAxis);
        if (norm > 0.001f) {
            primaryAxis = primaryAxis * (1.0f / norm);
        }
        
        secondaryAxis = cv::Point2f(-primaryAxis.y, primaryAxis.x);
        primaryAngle = std::atan2(primaryAxis.y, primaryAxis.x);
        
        mode = HAND_ORIENTATION_FALLBACK_AXIS;
    }
    
    // Convert to hand-local coordinates (rotation-invariant)
    cv::Point2f toHandLocal(const cv::Point2f& point) const {
        cv::Point2f relative = point - palmCenter;
        return cv::Point2f(
            relative.dot(secondaryAxis),  // Lateral position
            relative.dot(primaryAxis)     // Along-hand position
        );
    }
    
    // Convert hand-local to camera coordinates
    cv::Point2f toCamera(const cv::Point2f& local) const {
        return palmCenter + secondaryAxis * local.x + primaryAxis * local.y;
    }
    
    // Get hand-relative angle (stable representation)
    float getHandRelativeAngle(const cv::Point2f& point) const {
        cv::Point2f local = toHandLocal(point);
        return std::atan2(local.y, local.x);
    }
    
    // Reset state
    void reset() {
        palmCenter = cv::Point2f(-1, -1);
        wristMid = cv::Point2f(-1, -1);
        primaryAxis = cv::Point2f(0, 1);
        secondaryAxis = cv::Point2f(1, 0);
        primaryAngle = 0.0f;
        primaryAngleVelocity = 0.0f;
        angleHistory.clear();
        lastStableAngle = 0.0f;
        mode = HAND_ORIENTATION_WRIST_BASED;
    }
    
    // Check if frame is valid
    bool isValid() const {
        return palmCenter.x >= 0 && wristMid.x >= 0;
    }
};

enum FingerState {
    FINGER_STATE_FIST = 0,
    FINGER_STATE_PARTIAL = 1,
    FINGER_STATE_OPEN = 2
};

struct FingerIdentity {
    int id = -1;
    cv::Point2f rawTip;
    cv::Point2f displayTip = cv::Point2f(-1, -1);
    cv::Point2f lastTip;
    bool isDetected = false;
    int persistenceCounter = 0;
    float confidence = 0.0f;
    int framesSinceUpdate = 0;
    bool isLocked = false;
    int lockFrames = 0;
    int stableFrames = 0;
    
    // Hand-local representation (CRITICAL FOR STABILITY)
    cv::Point2f handLocalTip;          // Position in stable hand frame
    float handLocalAngle = 0.0f;       // Angle in hand frame
    float handLocalRadius = 0.0f;      // Normalized radius
    
    // Temporal coherence
    cv::Point2f handLocalVelocity;     // Velocity in hand-local space
    float angularVelocity = 0.0f;      // Angular velocity
    cv::Point2f velocityHistory[3];    // Recent velocities
    int velocityHistoryIndex = 0;
    
    // Locking and reassignment control
    int cooldownCounter = 0;           // Frames until reassignment allowed
    int consecutiveUpdates = 0;        // Frames with stable position
    
    // History for continuity
    std::deque<cv::Point2f> localPositionHistory;
    std::deque<float> localAngleHistory;
    
    void update(const cv::Point2f& newRawTip, const cv::Point2f& newHandLocalTip,
                float newHandLocalAngle, float newHandLocalRadius,
                const StableHandReferenceFrame& frame);
    void decay();
    void updatePositionOnly(const cv::Point2f& newRawTip, const cv::Point2f& newHandLocalTip,
                           float newHandLocalAngle, float newHandLocalRadius);
    
    float getReassignmentPenalty() const;
    bool shouldAllowReassignment(float newHandLocalAngle, float rotationDelta) const;
    float computeContinuityScore(const cv::Point2f& candidateLocalTip, 
                                 float candidateHandLocalAngle) const;
    
    // Update velocity in hand-local space
    void updateVelocity(const cv::Point2f& newHandLocalTip, float newHandLocalAngle);
    
    // Reset all history
    void resetHistory();
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
                    const cv::Point2f& wristMid,
                    float palmRadius, float avgFingerDistance, float handScale);
    
    bool validateContour(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter, float currentScale);
    
    bool isAnchored() const { return anchor.isAnchored; }
};

struct HandGeometryState {
    // Raw inputs (no feedback from smoothing)
    cv::Point2f rawPalmCenter = cv::Point2f(-1, -1);
    std::vector<cv::Point> rawContour;
    
    // Processed geometry (independent smoothing paths)
    cv::Point2f palmCenter = cv::Point2f(-1, -1);
    cv::Point2f wristMid = cv::Point2f(-1, -1);
    cv::Point2f wristLeft = cv::Point2f(-1, -1);
    cv::Point2f wristRight = cv::Point2f(-1, -1);
    float palmRadius = 0.0f;
    float thumbPinkyBaseWidth = 0.0f;
    
    // Smoothed outputs (separate paths, no cross-dependencies)
    cv::Point2f smoothedPalmCenter = cv::Point2f(-1, -1);
    cv::Point2f smoothedWristMid = cv::Point2f(-1, -1);
    cv::Point2f smoothedWristLeft = cv::Point2f(-1, -1);
    cv::Point2f smoothedWristRight = cv::Point2f(-1, -1);
    
    // Stable hand reference frame (CRITICAL)
    StableHandReferenceFrame handFrame;
    
    // Finger tracking (in hand-local space only)
    std::vector<FingerIdentity> fingerIdentities;
    std::vector<FingerIdentity> lastFrameFingers;
    
    // State tracking
    FingerState fingerState = FINGER_STATE_OPEN;
    int detectedFingerCount = 0;
    bool palmValid = false;
    int handValidityGraceCounter = 0;
    bool wasValidLastFrame = false;
    
    // Smoothing buffers (independent)
    std::deque<cv::Point2f> palmSmoothingBuffer;
    std::deque<cv::Point2f> wristLeftSmoothingBuffer;
    std::deque<cv::Point2f> wristRightSmoothingBuffer;
    
    // Fist detection
    std::vector<cv::Point> palmContour;
    
    // Wrist model state - PRIMARY DATA FLOW
    cv::Point2f thumbBase = cv::Point2f(-1, -1);
    cv::Point2f pinkyBase = cv::Point2f(-1, -1);
    cv::Point2f lastValidThumbBase = cv::Point2f(-1, -1);
    cv::Point2f lastValidPinkyBase = cv::Point2f(-1, -1);
    cv::Point2f lastValidWristLeft = cv::Point2f(-1, -1);
    cv::Point2f lastValidWristRight = cv::Point2f(-1, -1);
    cv::Point2f lastValidWristMid = cv::Point2f(-1, -1);
    float lastValidThumbPinkyWidth = 0.0f;
    
    // Geometric constraint helpers
    bool isPointInContour(const cv::Point2f& point) const;
    cv::Point2f projectPointToContourInterior(const cv::Point2f& point, const cv::Point2f& fallback) const;
    
    // STEP 1: Thumb & Pinky Base Extraction (from contour only)
    std::pair<cv::Point2f, cv::Point2f> findThumbPinkyBasesFromContour();
    
    // STEP 2-4: Wrist Computation Pipeline - NOW CONSUMES AUTHORITATIVE INPUT
    void computeWristFromBases(const cv::Point2f& thumbBase, const cv::Point2f& pinkyBase);
    
    // Anti-drift continuity
    void applyTemporalContinuity(const cv::Point2f& newPalmCenter, 
                                const cv::Point2f& lastPalmCenter,
                                float rotationAngle);
    
    void reset();
    
    // Independent processing paths
    void updateRawGeometry(const cv::Point2f& palm, const std::vector<cv::Point>& contour);
    void smoothPalmCenter();
    void smoothWristGeometry();  // Smooths left and right independently
    void updateHandReferenceFrame();
    
    // Finger processing (hand-local only)
    std::vector<FingerIdentity> detectFingerCandidates();
    void matchFingerIdentities(std::vector<FingerIdentity>& candidates);
    void updateFingerTracking();
    
    // Utility
    cv::Point2f projectToContour(const cv::Point2f& point) const;
    FingerState classifyFingerState() const;
    bool shouldRemainValid() const;
    void updateValidity(bool newValid);
    
    // Getters
    const StableHandReferenceFrame& getHandFrame() const { return handFrame; }
    bool isPalmValid() const { return palmValid; }
    const std::vector<FingerIdentity>& getFingerIdentities() const { return fingerIdentities; }
    FingerState getFingerState() const { return fingerState; }
    int getDetectedFingerCount() const { return detectedFingerCount; }
    cv::Point2f getPalmCenter() const { return smoothedPalmCenter; }
    cv::Point2f getWristMid() const { return smoothedWristMid; }
    cv::Point2f getWristLeft() const { return smoothedWristLeft; }
    cv::Point2f getWristRight() const { return smoothedWristRight; }
    const std::vector<cv::Point>& getPalmContour() const { return palmContour; }
    float getPalmRadius() const { return palmRadius; }
    cv::Point2f getThumbBase() const { return thumbBase; }
    cv::Point2f getPinkyBase() const { return pinkyBase; }
};


class GeometryUpdater {
private:
    HandGeometryState currentState;
    cv::Point2f lastPalmCenter = cv::Point2f(-1, -1);
    cv::Point2f lastWristMid = cv::Point2f(-1, -1);
    cv::Point2f lastThumbBase = cv::Point2f(-1, -1);
    cv::Point2f lastPinkyBase = cv::Point2f(-1, -1);
    float lastThumbPinkyWidth = 0.0f;
    
public:
    GeometryUpdater() {
        reset();
    }
    
    void reset() {
        currentState.reset();
        lastPalmCenter = cv::Point2f(-1, -1);
        lastWristMid = cv::Point2f(-1, -1);
        lastThumbBase = cv::Point2f(-1, -1);
        lastPinkyBase = cv::Point2f(-1, -1);
        lastThumbPinkyWidth = 0.0f;
    }
    
    // Main update - CONSUMES AUTHORITATIVE wrist geometry from PalmEstimator
    void updateGeometry(const cv::Point2f& rawPalmCenter,
                       const cv::Point2f& authoritativeWristMid,
                       const cv::Point2f& authoritativeWristLeft,
                       const cv::Point2f& authoritativeWristRight,
                       const std::vector<cv::Point>& constrainedContour);
    
    const HandGeometryState& getState() const { return currentState; }
    bool isPalmValid() const { return currentState.isPalmValid(); }
    cv::Point2f getPalmCenter() const { return currentState.getPalmCenter(); }
    cv::Point2f getWristMid() const { return currentState.getWristMid(); }
    cv::Point2f getWristLeft() const { return currentState.getWristLeft(); }
    cv::Point2f getWristRight() const { return currentState.getWristRight(); }
    
    // Helper methods to get specific finger tips
    cv::Point2f getThumbTip() const {
        const auto& fingers = currentState.getFingerIdentities();
        for (const auto& finger : fingers) {
            if (finger.isDetected && finger.id == 0) {
                return finger.displayTip;
            }
        }
        return cv::Point2f(-1, -1);
    }
    
    cv::Point2f getIndexTip() const {
        const auto& fingers = currentState.getFingerIdentities();
        for (const auto& finger : fingers) {
            if (finger.isDetected && finger.id == 1) {
                return finger.displayTip;
            }
        }
        return cv::Point2f(-1, -1);
    }
    
    cv::Point2f getMiddleFingerTip() const {
        const auto& fingers = currentState.getFingerIdentities();
        for (const auto& finger : fingers) {
            if (finger.isDetected && finger.id == 2) {
                return finger.displayTip;
            }
        }
        return cv::Point2f(-1, -1);
    }
    
    const std::vector<FingerIdentity>& getFingerIdentities() const { 
        return currentState.getFingerIdentities(); 
    }
    
    FingerState getFingerState() const { return currentState.getFingerState(); }
    int getDetectedFingerCount() const { return currentState.getDetectedFingerCount(); }
    
    const std::vector<cv::Point>& getPalmContour() const {
        return currentState.getPalmContour();
    }
    
    const StableHandReferenceFrame& getHandFrame() const { 
        return currentState.getHandFrame(); 
    }
    
    cv::Point2f getThumbBase() const { return currentState.getThumbBase(); }
    cv::Point2f getPinkyBase() const { return currentState.getPinkyBase(); }
};
#endif