#ifndef PALM_ESTIMATOR_H
#define PALM_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include "FingerAnalyzer.h"  // Now includes the prefixed constants

// PalmEstimator-specific constants (prefixed with PE_)
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
constexpr int PE_ANATOMICAL_CONSISTENCY_THRESHOLD = 3;

// Use the constants from FingerAnalyzer.h with their FA_ prefix
#define MIN_DEFECT_DEPTH_RATIO FA_MIN_DEFECT_DEPTH_RATIO
#define WRIST_FINGER_DISTANCE_RATIO FA_WRIST_FINGER_DISTANCE_RATIO
#define MAX_WRIST_DISTANCE_RATIO FA_MAX_WRIST_DISTANCE_RATIO
#define THUMB_PINKY_BASE_WIDTH_RATIO_MIN FA_THUMB_PINKY_BASE_WIDTH_RATIO_MIN
#define THUMB_PINKY_BASE_WIDTH_RATIO_MAX FA_THUMB_PINKY_BASE_WIDTH_RATIO_MAX
#define ANATOMICAL_CONFIDENCE_DECAY FA_ANATOMICAL_CONFIDENCE_DECAY

class PalmEstimator {
public:
    struct Result {
        cv::Point2f palm{-1,-1};
        cv::Point2f fingertip{-1,-1};
        cv::Point2f smoothedPalm{-1,-1};
        cv::Point2f smoothedFingertip{-1,-1};
        cv::Point2f thumbBase{-1,-1};
        cv::Point2f pinkyBase{-1,-1};
        bool handDetected = false;
        bool hasMotion = false;
        std::vector<cv::Point> contour;
        cv::Rect boundingBox;
        double area = 0;
        std::string status;
        
        float motionPercentage = 0;
        float aspectRatio = 0;
        float solidity = 0;
        float confidence = 0;
        
        int contoursFound = 0;
        int potentialHands = 0;
        
        float deltaStrength = 0;
        float handSizeScale = 1.0f;
        
        std::vector<cv::Point2f> fingerTips;
    };

    PalmEstimator() = default;
    
    Result detect(const cv::Mat& frame, const cv::Mat& motionMask, 
                  const cv::Mat& skinMask, float hsvConfidence);
    
    bool validatePalmShape(const std::vector<cv::Point>& contour, const cv::Point2f& palmCenter);
    float computeGeometryConsistencyScore(const std::vector<cv::Point>& contour, 
                                         const cv::Point2f& palmCenter, float palmRadius);
    bool couldBeHand(const std::vector<cv::Point>& contour, float& aspect, float& solidity);
    bool fitContourToModel(std::vector<cv::Point>& contour, 
                          cv::Point2f& palmCenter,
                          cv::Point2f& thumbBase,
                          cv::Point2f& pinkyBase,
                          float& handSizeScale);
    std::vector<cv::Point2f> detectFingerTips(const std::vector<cv::Point>& contour, 
                                             const cv::Point2f& palmCenter);
    
    void draw(cv::Mat& frame, const Result& result, 
              const cv::Mat& skinMask, const cv::Mat& motionMask);
};

#endif