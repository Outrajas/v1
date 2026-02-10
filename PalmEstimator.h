// PalmEstimator.h
#ifndef PALM_ESTIMATOR_H
#define PALM_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include "FingerAnalyzer.h"

// PalmEstimator-specific constants
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

// Use the constants from FingerAnalyzer.h with their FA_ prefix
#define MIN_DEFECT_DEPTH_RATIO FA_MIN_DEFECT_DEPTH_RATIO

class PalmEstimator {
public:
    struct Result {
        cv::Point2f palm{-1,-1};          // REQUIRED: contour-validated palm center
        cv::Point2f wristHint{-1,-1};     // NON-AUTHORITATIVE: GeometryUpdater ignores this
        
        // AUTHORITATIVE WRIST GEOMETRY (2-point line model)
        cv::Point2f wristLeft{-1,-1};
        cv::Point2f wristRight{-1,-1};
        cv::Point2f wristMid{-1,-1};
        
        bool handDetected = false;
        std::vector<cv::Point> contour;          // Raw HSV contour
        std::vector<cv::Point> constrainedContour; // Sculpted contour (Iteration 2)
        
        cv::Rect boundingBox;
        double area = 0;
        std::string status;
        
        float aspectRatio = 0;
        float solidity = 0;
        float confidence = 0;  // Wrist geometry confidence
        
        int contoursFound = 0;
        int potentialHands = 0;
        
        float handSizeScale = 1.0f;
        
        // Finger state info (filled by GeometryUpdater)
        FingerState fingerState = FINGER_STATE_OPEN;
        int detectedFingerCount = 0;
    };

    PalmEstimator() = default;
    
    Result detect(const cv::Mat& frame, const cv::Mat& motionMask, 
                  const cv::Mat& skinMask, float hsvConfidence);
    
    bool validatePalmShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter);
    float computeGeometryConsistencyScore(const std::vector<cv::Point>& contour, 
                                         const cv::Point2f& palmCenter, float palmRadius);
    bool couldBeHand(const std::vector<cv::Point>& contour, float& aspect, float& solidity);
    
    // PURE WRIST ESTIMATION - NO TRIMMING (Iteration 2 correction)
    float computeWristGeometry(const std::vector<cv::Point>& rawContour,
                             cv::Point2f& wristLeft,
                             cv::Point2f& wristRight,
                             cv::Point2f& wristMid);
    
    // ENHANCED SHAPE SCULPTING (Iteration 2)
    std::vector<cv::Point> constrainContourWithWrist(const std::vector<cv::Point>& rawContour,
                                                   const cv::Point2f& wristLeft,
                                                   const cv::Point2f& wristRight,
                                                   const cv::Point2f& wristMid,
                                                   float wristConfidence) const;
    
    // Palm center from constrained contour - NEVER FAILS
    bool fitContourToModel(std::vector<cv::Point>& contour, 
                          cv::Point2f& palmCenter,
                          float& handSizeScale);
    
    void draw(cv::Mat& frame, const Result& result, 
              const cv::Mat& skinMask, const cv::Mat& motionMask);
    
private:
    // Helper functions for contour geometry
    bool isPointInContour(const cv::Point2f& point, const std::vector<cv::Point>& contour) const;
    cv::Point2f projectPointToContourInterior(const cv::Point2f& point, 
                                              const std::vector<cv::Point>& contour,
                                              const cv::Point2f& fallback) const;
    
    // Wrist computation helpers
    std::pair<int, int> findThumbPinkyDefects(const std::vector<cv::Point>& contour,
                                             const std::vector<cv::Vec4i>& defects) const;
    
    // ITERATION 2: Shape sculpting helper functions
    std::vector<cv::Point> applyWristLineTrimming(const std::vector<cv::Point>& contour,
                                                const cv::Point2f& wristLeft,
                                                const cv::Point2f& wristRight,
                                                float wristConfidence) const;
    
    std::vector<cv::Point> applyLobeLimiting(const std::vector<cv::Point>& contour,
                                           float confidence) const;
    
    std::vector<cv::Point> applyWidthConsistencyCheck(const std::vector<cv::Point>& contour,
                                                    const cv::Point2f& wristLeft,
                                                    const cv::Point2f& wristRight,
                                                    float confidence) const;
    
    std::vector<cv::Point> applyCalibrationBiasedRefinement(const std::vector<cv::Point>& contour,
                                                          float confidence) const;
};

#endif