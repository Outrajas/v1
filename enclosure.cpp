// enclosure.cpp – enclosure analysis, contour extraction, shape check
#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace cv;
using namespace std;

constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;
constexpr int GRID_COLS = 160;
constexpr int GRID_ROWS = 120;
constexpr int CELL_W = FRAME_WIDTH / GRID_COLS;
constexpr int CELL_H = FRAME_HEIGHT / GRID_ROWS;
constexpr float PI = 3.14159265f;

struct RayResult {
    Point2f start;
    Point2f end;
    bool hitBarrier;
    float support;
};

struct EnclosureResult {
    float enclosureScore;
    float supportScore;
    float meshContinuity;
    float confidence;
    vector<RayResult> rays;
    vector<Point2f> anchors;
};

EnclosureResult analyzeEnclosure(const vector<Point>& interiorCells,
                                 const Mat& barrierMap,
                                 const Rect& roi) {
    EnclosureResult res;
    res.enclosureScore = 1.0f;
    res.supportScore = 1.0f;
    res.meshContinuity = 1.0f;
    res.confidence = 1.0f;

    if (interiorCells.empty()) return res;

    // --- Compute integral image for fast support computation ---
    Mat integralImg;
    integral(barrierMap, integralImg, CV_32S);

    auto getRectSum = [&](int x1, int y1, int x2, int y2) -> int {
        x1 = std::max(0, x1); y1 = std::max(0, y1);
        x2 = std::min(barrierMap.cols - 1, x2);
        y2 = std::min(barrierMap.rows - 1, y2);
        if (x1 > x2 || y1 > y2) return 0;
        int x2p = x2 + 1, y2p = y2 + 1;
        int x1p = x1, y1p = y1; 
        return integralImg.at<int>(y2p, x2p) - integralImg.at<int>(y1p, x2p)
             - integralImg.at<int>(y2p, x1p) + integralImg.at<int>(y1p, x1p);
    };

    // --- Anchor selection ---
    vector<Point2f> anchors;
    {
        int minX = INT_MAX, maxX = 0, minY = INT_MAX, maxY = 0;
        Point2f north(0, (float)INT_MAX), south(0,0), east(0,0), west((float)INT_MAX,0);
        for (const auto& p : interiorCells) {
            if (p.y < minY) { minY = p.y; north = Point2f((float)p.x, (float)p.y); }
            if (p.y > maxY) { maxY = p.y; south = Point2f((float)p.x, (float)p.y); }
            if (p.x < minX) { minX = p.x; west  = Point2f((float)p.x, (float)p.y); }
            if (p.x > maxX) { maxX = p.x; east  = Point2f((float)p.x, (float)p.y); }
        }
        anchors.push_back(north); anchors.push_back(south); anchors.push_back(east); anchors.push_back(west);
        float cx=0, cy=0;
        for (const auto& p : interiorCells) { cx += p.x; cy += p.y; }
        anchors.push_back(Point2f(cx/interiorCells.size(), cy/interiorCells.size()));

        if (interiorCells.size() > anchors.size()) {
            int step = max(1, (int)(interiorCells.size() / (32 - anchors.size())));
            for (size_t i = 0; i < interiorCells.size(); i += step) {
                anchors.push_back(Point2f((float)interiorCells[i].x, (float)interiorCells[i].y));
                if (anchors.size() >= 32) break;
            }
        }
    }

    // --- Ray casting with integral image support ---
    const int numDirs = 16;
    const float angleStep = 2*PI / numDirs;
    int totalRays = 0, escapedRays = 0;
    float totalSupport = 0;
    int supportCount = 0;
    vector<tuple<Point2f,float,Point2f>> hitPoints;

    for (const Point2f& anchor : anchors) {
        for (int i = 0; i < numDirs; ++i) {
            float angle = i * angleStep;
            float dx = cosf(angle), dy = sinf(angle);
            Point2f cur = anchor;
            bool hitBarrier = false;
            const int maxSteps = 200;
            for (int s = 0; s < maxSteps; s++) {
                cur.x += dx; cur.y += dy;
                if (!roi.contains(Point((int)cur.x, (int)cur.y))) break;
                if (cur.x < 0 || cur.x >= FRAME_WIDTH || cur.y < 0 || cur.y >= FRAME_HEIGHT) break;
                if (barrierMap.at<uchar>((int)cur.y, (int)cur.x) != 0) {
                    hitBarrier = true;
                    break;
                }
            }
            totalRays++;
            RayResult ray;
            ray.start = anchor; ray.end = cur; ray.hitBarrier = hitBarrier;

            int r = 5;
            int x = (int)cur.x, y = (int)cur.y;
            int x1 = x - r, y1 = y - r, x2 = x + r, y2 = y + r;
            int sum = getRectSum(x1, y1, x2, y2);
            int count = (std::min(x2, barrierMap.cols-1) - std::max(x1, 0) + 1) *
                        (std::min(y2, barrierMap.rows-1) - std::max(y1, 0) + 1);
            if (count > 0) {
                ray.support = (float)sum / count;
                totalSupport += ray.support;
                supportCount++;
            } else ray.support = 0;

            if (!hitBarrier) escapedRays++;
            res.rays.push_back(ray);
            if (hitBarrier) hitPoints.emplace_back(anchor, angle, cur);
        }
    }

    int trappedRays = totalRays - escapedRays;
    res.enclosureScore = (totalRays > 0) ? (float)trappedRays / totalRays : 1.0f;
    res.supportScore = (supportCount > 0) ? (totalSupport / supportCount) : 0;

    {
        vector<float> angles;
        for (const auto& [a, ang, ep] : hitPoints) angles.push_back(ang);
        if (angles.size() >= 2) {
            sort(angles.begin(), angles.end());
            float prev = angles.back() - 2*PI;
            float totalGap = 0;
            for (float a : angles) { totalGap += a - prev; prev = a; }
            float avgGap = totalGap / angles.size();
            float expectedGap = 2*PI / numDirs;
            res.meshContinuity = std::min(1.0f, expectedGap / (avgGap + 1e-5f));
        } else res.meshContinuity = 0.5f;
    }

    res.confidence = 0.4f * res.enclosureScore + 0.3f * res.supportScore + 0.3f * res.meshContinuity;
    res.anchors = anchors;
    return res;
}

vector<Point> extractOuterContour(const Mat& interiorMask, const Mat& barrierMap) {
    Mat spatialMask = Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
    for (int r = 0; r <= GRID_ROWS; r++) {
        for (int c = 0; c <= GRID_COLS; c++) {
            if (interiorMask.at<uchar>(r,c) > 0) {
                int x = min(c * CELL_W, FRAME_WIDTH - 1);
                int y = min(r * CELL_H, FRAME_HEIGHT - 1);
                int w = min(CELL_W, FRAME_WIDTH - x);
                int h = min(CELL_H, FRAME_HEIGHT - y);
                if (w > 0 && h > 0) spatialMask(Rect(x, y, w, h)).setTo(255);
            }
        }
    }

    Mat dilated;
    // Increased structuring element size slightly to ensure robust contour merging
    dilate(spatialMask, dilated, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

    vector<vector<Point>> contours;
    findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {};

    double maxArea = 0;
    int bestIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) { maxArea = area; bestIdx = i; }
    }
    if (bestIdx < 0) return {};

    vector<Point> rawContour = contours[bestIdx];
    vector<Point> refined;
    for (const auto& pt : rawContour) {
        bool found = false;
        for (int dy = -2; dy <= 2 && !found; dy++) {
            for (int dx = -2; dx <= 2 && !found; dx++) {
                int nx = pt.x + dx, ny = pt.y + dy;
                if (nx >= 0 && nx < barrierMap.cols && ny >= 0 && ny < barrierMap.rows) {
                    if (barrierMap.at<uchar>(ny, nx) != 0) {
                        refined.push_back(Point(nx, ny));
                        found = true;
                    }
                }
            }
        }
        if (!found) refined.push_back(pt);
    }

    vector<Point> approx;
    approxPolyDP(refined, approx, 2.0, true);
    return approx;
}

bool isHandShaped(const vector<Point>& contour) {
    if (contour.size() < 15) return false;
    double area = cv::contourArea(contour);
    if (area < 500) return false;

    Rect bb = boundingRect(contour);
    float aspect = (float)bb.width / std::max(1, bb.height);
    if (aspect < 0.3f || aspect > 3.0f) return false;

    double perimeter = arcLength(contour, true);
    double circularity = (perimeter * perimeter) / (4.0 * CV_PI * area);
    if (circularity < 1.5) return false;

    vector<Point> hull;
    convexHull(contour, hull);
    double hullArea = cv::contourArea(hull);
    if (hullArea < 1.0) return false;
    double convexity = area / hullArea;
    if (convexity > 0.95f) return false;

    double extent = area / (double)(bb.width * bb.height);
    if (extent < 0.25) return false;

    return true;
}