// edge_extraction.cpp – 5‑channel consensus with optional morphology & blur
#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;

// ── Local compile‑time constants ──────────────────────────────────────────
constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;
constexpr int ROI_MARGIN = 15;
constexpr double NOISE_FLOOR = 15.0;
constexpr int REQUIRED_VOTES = 5;
constexpr int MIN_BLOB_AREA = 10;

// ── Mask processing parameters (tweak these) ──────────────────────────────
constexpr int DILATE_SIZE = 1;      // 0=off, 1=3x3, 2=5x5, 3=7x7
constexpr int DILATE_ITER = 1;      // number of passes
constexpr int ERODE_SIZE  = 1;      // 0=off, 1=3x3, 2=5x5, 3=7x7
constexpr int ERODE_ITER  = 1;
constexpr int GAUSS_SIGMA = 1;      // 0=off, >0 applies blur (then re‑threshold)
constexpr int GAUSS_KSIZE = 2;      // 1=3x3, 2=5x5 (only if GAUSS_SIGMA>0)

extern Mat globalReferenceVisual;
extern Mat globalStructuralBarrier;
extern Mat globalRawStacked;

static void getSobelMagnitude(const Mat& channel, Mat& magOut) {
    Mat grad_x, grad_y;
    Sobel(channel, grad_x, CV_32F, 1, 0, 3);
    Sobel(channel, grad_y, CV_32F, 0, 1, 3);
    add(abs(grad_x), abs(grad_y), magOut);
}

void extractEdges(const Mat& frame, const Rect& roi, Mat& refVisualOut, Mat& structuralBarrierOut) {
    Rect roiPadded = roi;
    roiPadded.x = std::max(0, roi.x - ROI_MARGIN);
    roiPadded.y = std::max(0, roi.y - ROI_MARGIN);
    roiPadded.width  = std::min(FRAME_WIDTH  - roiPadded.x, roi.width  + 2*ROI_MARGIN);
    roiPadded.height = std::min(FRAME_HEIGHT - roiPadded.y, roi.height + 2*ROI_MARGIN);

    Mat roiFrame = frame(roiPadded);
    Mat hsv, ycrcb;
    cvtColor(roiFrame, hsv, COLOR_BGR2HSV);
    cvtColor(roiFrame, ycrcb, COLOR_BGR2YCrCb);

    vector<Mat> bgr(3), hsv_channels(3), ycrcb_channels(3);
    split(roiFrame, bgr);
    split(hsv, hsv_channels);
    split(ycrcb, ycrcb_channels);

    vector<Mat> mags(5);
    getSobelMagnitude(bgr[0], mags[0]);
    getSobelMagnitude(bgr[1], mags[1]);
    getSobelMagnitude(bgr[2], mags[2]);
    getSobelMagnitude(ycrcb_channels[0], mags[3]);
    getSobelMagnitude(hsv_channels[2], mags[4]);

    Mat rawStacked = Mat::zeros(roiFrame.size(), CV_32F);
    Mat consensusCount = Mat::zeros(roiFrame.size(), CV_8UC1);
    Mat tempVote;
    for (int i = 0; i < 5; i++) {
        add(rawStacked, mags[i], rawStacked);
        threshold(mags[i], tempVote, NOISE_FLOOR, 1, THRESH_BINARY);
        tempVote.convertTo(tempVote, CV_8UC1);
        add(consensusCount, tempVote, consensusCount);
    }

    Mat binaryMask;
    threshold(consensusCount, binaryMask, REQUIRED_VOTES - 1, 255, THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8UC1);  // ensure 0/255

    // ── Optional morphological processing ──────────────────────────────────
    // Dilation
    if (DILATE_SIZE > 0) {
        int ksize = 2 * DILATE_SIZE + 1;   // 3,5,7
        Mat kernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
        for (int i = 0; i < DILATE_ITER; ++i)
            dilate(binaryMask, binaryMask, kernel);
    }

    // Erosion
    if (ERODE_SIZE > 0) {
        int ksize = 2 * ERODE_SIZE + 1;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
        for (int i = 0; i < ERODE_ITER; ++i)
            erode(binaryMask, binaryMask, kernel);
    }

    // Gaussian blur (with re‑threshold to keep binary)
    if (GAUSS_SIGMA > 0 && GAUSS_KSIZE > 0) {
        int ksize = 2 * GAUSS_KSIZE + 1;   // 3 or 5
        GaussianBlur(binaryMask, binaryMask, Size(ksize, ksize), (double)GAUSS_SIGMA);
        threshold(binaryMask, binaryMask, 128, 255, THRESH_BINARY);
    }

    // ── Connected-component filtering ─────────────────────────────────────
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(binaryMask, labels, stats, centroids, 8, CV_32S);
    Mat cleanMask = Mat::zeros(roiFrame.size(), CV_8UC1);
    for (int i = 1; i < nLabels; i++) {
        if (stats.at<int>(i, CC_STAT_AREA) >= MIN_BLOB_AREA) {
            cleanMask.setTo(255, labels == i);
        }
    }

    // ── Visualisation (unchanged) ──────────────────────────────────────────
    Mat localVisual = Mat::zeros(roiFrame.size(), CV_8UC3);
    for (int y = 0; y < roiFrame.rows; y++) {
        const float* mB = mags[0].ptr<float>(y);
        const float* mG = mags[1].ptr<float>(y);
        const float* mR = mags[2].ptr<float>(y);
        const float* mY = mags[3].ptr<float>(y);
        const float* mV = mags[4].ptr<float>(y);
        Vec3b* vis = localVisual.ptr<Vec3b>(y);
        for (int x = 0; x < roiFrame.cols; x++) {
            float vals[5] = {mB[x], mG[x], mR[x], mY[x], mV[x]};
            int idx = (int)(std::max_element(vals, vals+5) - vals);
            uchar intensity = saturate_cast<uchar>(vals[idx] * 0.25f);
            if (intensity > 16) {
                switch (idx) {
                    case 0: vis[x] = Vec3b(intensity, 0, 0); break;
                    case 1: vis[x] = Vec3b(0, intensity, 0); break;
                    case 2: vis[x] = Vec3b(0, 0, intensity); break;
                    case 3: vis[x] = Vec3b(0, intensity, intensity); break;
                    case 4: vis[x] = Vec3b(intensity, 0, intensity); break;
                }
            }
        }
    }

    refVisualOut.setTo(Scalar(0,0,0));
    structuralBarrierOut.setTo(0);
    localVisual.copyTo(refVisualOut(roiPadded));
    cleanMask.copyTo(structuralBarrierOut(roiPadded));

    Mat raw8;
    rawStacked.convertTo(raw8, CV_8U, 0.25);
    globalRawStacked.setTo(0);
    raw8.copyTo(globalRawStacked(roiPadded));
}