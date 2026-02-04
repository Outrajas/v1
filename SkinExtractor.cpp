#include "SkinExtractor.h"
#include <iostream>
#include <algorithm>

cv::Mat SkinExtractor::detectSkin(const cv::Mat& frame, float& outConfidence, bool isCalibrated, 
                                 const cv::Scalar& calibratedLower, const cv::Scalar& calibratedUpper,
                                 bool geometryValid) {
    cv::Mat hsv, skinMask;
    
    cv::Scalar lower = isCalibrated ? calibratedLower : HSV_LOWER;
    cv::Scalar upper = isCalibrated ? calibratedUpper : HSV_UPPER;
    
    bool hsvRelaxed = hsvIsRelaxed;
    
    static int relaxationFrames = 0;
    if (hsvRelaxed && geometryValid) {
        relaxationFrames = std::min(10, relaxationFrames + 1);
    } else {
        relaxationFrames = std::max(0, relaxationFrames - 1);
    }
    
    float relaxationFactor = 1.0f + (HSV_ADAPTIVE_TOLERANCE * (relaxationFrames / 10.0f));
    
    cv::Scalar relaxedLower = lower * (1.0f / relaxationFactor);
    cv::Scalar relaxedUpper = upper * relaxationFactor;
    
    relaxedLower[0] = std::max(0.0, relaxedLower[0]);
    relaxedUpper[0] = std::min(180.0, relaxedUpper[0]);
    
    for (int i = 1; i < 3; i++) {
        relaxedLower[i] = std::max(0.0, relaxedLower[i]);
        relaxedUpper[i] = std::min(255.0, relaxedUpper[i]);
    }
    
    cv::Mat blurred;
    cv::blur(frame, blurred, cv::Size(3, 3));
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, relaxedLower, relaxedUpper, skinMask);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(skinMask, skinMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(skinMask, skinMask, cv::MORPH_OPEN, kernel);
    
    int skinPixels = cv::countNonZero(skinMask);
    int dynamicMinSkin = 800 / 2;
    
    bool hsvValid = skinPixels >= dynamicMinSkin;
    outConfidence = geometryValid ? std::max(0.3f, static_cast<float>(skinPixels) / dynamicMinSkin) : 0.0f;
    
    updateHSVState(hsvValid, geometryValid, skinPixels, dynamicMinSkin);
    
    return skinMask;
}

void SkinExtractor::updateHSVState(bool hsvValid, bool geometryValid, int skinPixels, int dynamicMinSkin) {
    if (hsvValid) {
        hsvFailureCounter = 0;
        hsvIsRelaxed = false;
        hsvConfidence = 1.0f;
    } else if (geometryValid) {
        hsvFailureCounter++;
        if (hsvFailureCounter >= HSV_VALID_GEOMETRY_GRACE_FRAMES) {
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