// globals.cpp – Constants, structs, and global state definitions
#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
using namespace cv;
using namespace std;

// Local constants (compile‑time, not shared)
constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;
constexpr int GRID_COLS = 160;
constexpr int GRID_ROWS = 120;
constexpr int CELL_W = FRAME_WIDTH / GRID_COLS;
constexpr int CELL_H = FRAME_HEIGHT / GRID_ROWS;

// Struct definitions 
struct TrackedObject {
    Mat prevInteriorMask;
    vector<vector<Point>> prevContours;
    Point2f centroid;
    float area;
    bool valid = false;
};

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

struct TopologyOutput {
    Mat interiorMask;
    vector<Point> surfaceEdges;
    vector<Point> interiorCells;
    vector<pair<Point, Point>> bfsGraph;
    bool success = false;
};

// Global state definitions
Mat globalReferenceVisual;
Mat globalStructuralBarrier;
Mat globalRawStacked;
Rect currentROI(160, 120, 320, 240);
Point2f current_seed(FRAME_WIDTH*0.5f, FRAME_HEIGHT*0.5f);
Point2f roiCenter(FRAME_WIDTH*0.5f, FRAME_HEIGHT*0.5f);
Point2f rawMeasuredCentroid(FRAME_WIDTH*0.5f, FRAME_HEIGHT*0.5f);
bool trackingRobust = false;
int lostFrames = 0;
int fps = 0, frameCount = 0;
chrono::steady_clock::time_point lastTime = chrono::steady_clock::now();
TrackedObject trackedObject;
TopologyOutput lastTopology;
EnclosureResult lastEnclosure;
float gLeakConfidence = 1.0f;

void drawGrid(Mat& img) {
    Scalar c(40,40,40);
    for (int i=0; i<=GRID_COLS; i+=4)
        line(img, Point(i*CELL_W,0), Point(i*CELL_W,FRAME_HEIGHT), c, 1);
    for (int j=0; j<=GRID_ROWS; j+=4)
        line(img, Point(0,j*CELL_H), Point(FRAME_WIDTH,j*CELL_H), c, 1);
}