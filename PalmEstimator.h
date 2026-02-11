// PalmEstimator.h
#ifndef PALM_ESTIMATOR_H
#define PALM_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <tuple>
#include "FingerAnalyzer.h"

constexpr int PE_MIN_HAND_AREA = 800;
constexpr int PE_MAX_HAND_AREA = 80000;
constexpr float PE_DISTANCE_LOWER_BOUND = 0.65f;
constexpr float PE_DISTANCE_UPPER_BOUND = 1.5f;
constexpr int PE_MAX_FAILURE_FRAMES = 5;
constexpr float PE_MAX_PALM_CENTER_JUMP = 100.0f;
constexpr float PE_CONTOUR_RADIAL_VARIANCE_MAX = 0.35f;
constexpr float PE_CONTOUR_COMPACTNESS_MIN = 0.5f;
constexpr int PE_MIN_VALID_CONVEXITY_DEFECTS = 2;
constexpr float PE_MAX_SYMMETRY_SCORE = 0.85f;

// Hysteresis and freeze constants
constexpr float FREEZE_CONFIDENCE_ENTRY_THRESHOLD = 0.6f;
constexpr float FREEZE_CONFIDENCE_EXIT_THRESHOLD = 0.4f;
constexpr int FREEZE_FRAMES_REQUIRED = 10;
constexpr int UNFREEZE_FRAMES_REQUIRED = 5;
constexpr float DAMPED_ALPHA = 0.1f;
constexpr float BASE_SMOOTHING_ALPHA = 0.3f;
constexpr float ADAPTIVE_ALPHA_RANGE = 0.5f;

// Temporal confidence
constexpr float VELOCITY_CAP = 50.0f;

#define MIN_DEFECT_DEPTH_RATIO FA_MIN_DEFECT_DEPTH_RATIO

struct ArcAnalysis {
    std::pair<int, int> arc;
    float fingerScore;
    float armScore;
    float palmScore;
    float arcLength;
};

enum ArcClass {
    ARC_PALM,
    ARC_FINGER,
    ARC_ARM,
    ARC_UNKNOWN
};

struct ArcSegment {
    std::pair<int, int> indices;
    ArcClass classification;
    float confidence;
    cv::Point2f centroid;
    float avgDistance;
    float arcLength;
    float curvature;
    bool isProtected;
    int parentArcIdx;
    std::vector<int> childArcIndices;
    std::vector<int> keptPointIndices;
    float shortenRatio;
    bool isSevered;
    std::pair<int, int> severBoundary;
};

enum FreezeState {
    FREEZE_STATE_FREE,
    FREEZE_STATE_DAMPED,
    FREEZE_STATE_FROZEN
};

class PalmEstimator {
public:
    struct Result {
        cv::Point2f palm{-1,-1};
        cv::Point2f wristHint{-1,-1};
        cv::Point2f wristLeft{-1,-1};
        cv::Point2f wristRight{-1,-1};
        cv::Point2f wristMid{-1,-1};
        bool handDetected = false;
        std::vector<cv::Point> contour;
        std::vector<cv::Point> constrainedContour;
        cv::Rect boundingBox;
        double area = 0;
        std::string status;
        float aspectRatio = 0;
        float solidity = 0;
        float confidence = 0;
        int contoursFound = 0;
        int potentialHands = 0;
        float handSizeScale = 1.0f;
        FingerState fingerState = FINGER_STATE_OPEN;
        int detectedFingerCount = 0;
        
        // NEW: Debug and state exports
        FreezeState freezeState = FREEZE_STATE_FREE;
        int freezeCounter = 0;
        int unfreezeCounter = 0;
        float effectiveAlpha = BASE_SMOOTHING_ALPHA;
        float temporalConfidence = 1.0f;
        cv::Point2f rawWristLeft{-1,-1};
        cv::Point2f rawWristRight{-1,-1};
    };

    PalmEstimator();
    
    Result detect(const cv::Mat& frame, const cv::Mat& motionMask, 
                  const cv::Mat& skinMask, float hsvConfidence);
    
    bool validatePalmShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter);
    float computeGeometryConsistencyScore(const std::vector<cv::Point>& contour, 
                                         const cv::Point2f& palmCenter, float palmRadius);
    bool couldBeHand(const std::vector<cv::Point>& contour, float& aspect, float& solidity);
    
    float computeWristGeometry(const std::vector<cv::Point>& rawContour,
                             cv::Point2f& wristLeft,
                             cv::Point2f& wristRight,
                             cv::Point2f& wristMid);
    
    std::vector<cv::Point> constrainContourWithWrist(const std::vector<cv::Point>& rawContour,
                                                   const cv::Point2f& wristLeft,
                                                   const cv::Point2f& wristRight,
                                                   const cv::Point2f& wristMid,
                                                   float wristConfidence) const;
    
    bool fitContourToModel(std::vector<cv::Point>& contour, 
                          cv::Point2f& palmCenter,
                          float& handSizeScale);
    
    void draw(cv::Mat& frame, const Result& result, 
              const cv::Mat& skinMask, const cv::Mat& motionMask);

private:
    // Iteration 3: Structural Confidence Model
    struct StructuralConfidence {
        float palmCoreConfidence;
        float wristSpanConfidence;
        float contourCompactness;
        float armIntrusionConfidence;
        float temporalConfidence;
        float globalScore;
    };

    // Persistent smoothing state - NOW SINGLE AUTHORITY
    cv::Point2f m_lastSmoothedWristLeft;
    cv::Point2f m_lastSmoothedWristRight;
    cv::Point2f m_lastStableWristLeft;
    cv::Point2f m_lastStableWristRight;
    cv::Point2f m_lastSmoothedPalm;
    std::deque<cv::Point2f> m_palmSmoothingBuffer;
    
    float       m_lastAngle;
    float       m_lastStableAngle;
    int         m_stableFrameCount;
    float       m_lastAngularVelocity;
    bool        m_hasInitialState;
    
    // Hysteresis state - NEW
    FreezeState m_currentFreezeState;
    int         m_freezeCounter;
    int         m_unfreezeCounter;
    bool        m_freezeActive;
    cv::Point2f m_frozenWristLeft;
    cv::Point2f m_frozenWristRight;
    cv::Point2f m_frozenWristMid;
    cv::Point2f m_lastRawWristLeft;
    cv::Point2f m_lastRawWristRight;

    // Tunable constants
    const float m_baseAlpha = BASE_SMOOTHING_ALPHA;
    const float m_stableAngleThreshold = 0.1f;
    const int   m_stableFramesRequired = 6;
    const float m_idleVelocityThreshold = 0.08f;
    const float m_idleAlphaBoost = 1.15f;
    const float m_maxAlpha = 0.65f;

    // Helper functions for Structural Confidence
    StructuralConfidence computeStructuralConfidence(
        const std::vector<cv::Point>& contour,
        const cv::Point2f& palmCenter,
        const cv::Point2f& wristLeft,
        const cv::Point2f& wristRight,
        const cv::Point2f& wristMid,
        const cv::Point2f& thumbBase,
        const cv::Point2f& pinkyBase,
        float palmRadius,
        const cv::Point2f& lastRawWristLeft,
        const cv::Point2f& lastRawWristRight);

    float computePalmCoreConfidence(const std::vector<cv::Point>& contour,
                                   const cv::Point2f& palmCenter,
                                   float palmRadius) const;

    float computeWristSpanConfidence(float wristWidth, float thumbPinkyWidth) const;

    float computeContourCompactness(const std::vector<cv::Point>& contour) const;

    float computeArmIntrusionConfidence(const std::vector<cv::Point>& contour,
                                       const cv::Point2f& palmCenter,
                                       const cv::Point2f& wristMid,
                                       float palmRadius) const;
    
    // NEW: Temporal confidence
    float computeTemporalConfidence(const cv::Point2f& currentRawLeft,
                                   const cv::Point2f& currentRawRight,
                                   const cv::Point2f& lastRawLeft,
                                   const cv::Point2f& lastRawRight) const;

    // NEW: Hysteresis state machine
    void updateFreezeState(float globalScore, Result& result);
    
    // NEW: Centralized palm smoothing
    cv::Point2f smoothPalm(const cv::Point2f& rawPalm);

    // Smoothing and stabilization - NOW AUTHORITATIVE
    void applyExponentialSmoothing(cv::Point2f& smoothedLeft,
                                  cv::Point2f& smoothedRight,
                                  const cv::Point2f& rawLeft,
                                  const cv::Point2f& rawRight,
                                  float alpha,
                                  bool skipSmoothing);

    void applyProportionalDamping(cv::Point2f& left,
                                 cv::Point2f& right,
                                 float globalScore);

    void updateStableAnchors(const cv::Point2f& left,
                            const cv::Point2f& right,
                            float globalScore);

    // Existing helper functions
    bool isPointInContour(const cv::Point2f& point, const std::vector<cv::Point>& contour) const;
    cv::Point2f projectPointToContourInterior(const cv::Point2f& point, 
                                              const std::vector<cv::Point>& contour,
                                              const cv::Point2f& fallback) const;
    
    std::pair<int, int> findThumbPinkyDefects(const std::vector<cv::Point>& contour,
                                             const std::vector<cv::Vec4i>& defects) const;
    
    std::vector<ArcSegment> classifyArcsByStructure(
        const std::vector<cv::Point>& contour,
        const std::vector<ArcAnalysis>& arcAnalyses,
        const cv::Point2f& palmCenter,
        float palmRadius) const;
    
    void buildContourSkeleton(
        const std::vector<cv::Point>& contour,
        const cv::Point2f& palmCenter,
        float palmRadius,
        std::vector<ArcSegment>& structures) const;
    
    void shortenArcSegment(ArcSegment& segment, float shortenRatio, 
                          const std::vector<cv::Point>& contour) const;
    
    void severArcSegment(ArcSegment& segment, const std::vector<cv::Point>& contour) const;
    
    std::vector<std::pair<int, int>> computeContourDerivedSkeleton(
        const std::vector<cv::Point>& contour,
        const cv::Point2f& palmCenter,
        float palmRadius) const;
    
    std::tuple<float, float, float> analyzeSkeletalArc(
        const std::vector<cv::Point>& contour,
        const std::pair<int, int>& arc,
        const cv::Point2f& palmCenter,
        float palmRadius) const;
    
    std::vector<ArcSegment> applyWristLineTrimming(const std::vector<cv::Point>& contour,
                                                const cv::Point2f& wristLeft,
                                                const cv::Point2f& wristRight,
                                                float wristConfidence,
                                                std::vector<ArcSegment>& arcSegments) const;
    
    std::vector<ArcSegment> applyLobeLimiting(const std::vector<cv::Point>& contour,
                                            float confidence,
                                            std::vector<ArcSegment>& arcSegments,
                                            const cv::Point2f& palmCenter,
                                            float palmRadius) const;
    
    std::vector<ArcSegment> applyWidthConsistencyCheck(const std::vector<cv::Point>& contour,
                                                     const cv::Point2f& wristLeft,
                                                     const cv::Point2f& wristRight,
                                                     float confidence,
                                                     std::vector<ArcSegment>& arcSegments) const;
    
    std::vector<ArcSegment> applyCalibrationBiasedRefinement(const std::vector<cv::Point>& contour,
                                                           float confidence,
                                                           std::vector<ArcSegment>& arcSegments) const;
    
    std::vector<cv::Point> flattenArcSegmentsToContour(const std::vector<cv::Point>& originalContour,
                                                     const std::vector<ArcSegment>& arcSegments) const;
    
    // NEW: Debug overlay helpers
    void drawDebugStructures(cv::Mat& frame, 
                           const std::vector<cv::Point>& contour,
                           const std::vector<ArcSegment>& arcSegments,
                           const cv::Point2f& palmCenter) const;
    void drawWristOverlay(cv::Mat& frame, const Result& result) const;
    void drawConfidenceOverlay(cv::Mat& frame, const Result& result) const;
    void drawFreezeOverlay(cv::Mat& frame, const Result& result) const;
};

#endif