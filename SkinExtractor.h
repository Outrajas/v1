#ifndef SKIN_EXTRACTOR_H
#define SKIN_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

constexpr float HSV_ADAPTIVE_TOLERANCE = 0.25f;
constexpr int HSV_VALID_GEOMETRY_GRACE_FRAMES = 5;

class SkinExtractor {
private:
    cv::Scalar HSV_LOWER = cv::Scalar(0, 20, 70);
    cv::Scalar HSV_UPPER = cv::Scalar(25, 180, 255);
    
    bool hsvIsRelaxed = false;
    float hsvConfidence = 1.0f;
    int hsvFailureCounter = 0;
    
public:
    SkinExtractor() = default;
    
    cv::Mat detectSkin(const cv::Mat& frame, float& outConfidence, bool isCalibrated, 
                      const cv::Scalar& calibratedLower, const cv::Scalar& calibratedUpper,
                      bool geometryValid);
    
    bool isHSVRelaxed() const { return hsvIsRelaxed; }
    float getHSVConfidence() const { return hsvConfidence; }
    void updateHSVState(bool hsvValid, bool geometryValid, int skinPixels, int dynamicMinSkin);
};

#endif