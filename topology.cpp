// topology.cpp – Optimized BFS interior/surface segmentation
#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
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
constexpr float MAX_HAND_W = 300.f;
constexpr float MAX_HAND_H = 360.f;

struct TopologyOutput {
    Mat interiorMask;
    vector<Point> surfaceEdges;
    vector<Point> interiorCells;
    vector<pair<Point, Point>> bfsGraph;
    bool success = false;
};

TopologyOutput runTopologyBFS(const Mat& barrierMap, const Rect& roi, Point2f predictedSeed,
                              const Mat& prevMask, bool havePrev, float leakConf) {
    TopologyOutput out;
    if (roi.area() <= 0) return out;

    float baseFactor = 2.0f;
    if (leakConf > 0.85f)      baseFactor = 1.8f;
    else if (leakConf > 0.6f)  baseFactor = 1.3f;
    else                       baseFactor = 1.05f;

    int maxCells = (int)(MAX_HAND_W * MAX_HAND_H / (CELL_W * CELL_H)) * baseFactor;
    if (havePrev && !prevMask.empty()) {
        float prevAreaCells = 0;
        for (int r=0; r<=GRID_ROWS; r++)
            for (int c=0; c<=GRID_COLS; c++)
                if (prevMask.at<uchar>(r,c) > 0) prevAreaCells++;
        maxCells = std::min(maxCells, (int)(prevAreaCells * baseFactor * 1.8f));
        maxCells = std::max(50, maxCells);
    }

    enum NodeState : uint8_t { UNKNOWN = 0, INTERIOR, SURFACE_CANDIDATE, EXTERIOR };
    vector<vector<NodeState>> vState(GRID_ROWS+1, vector<NodeState>(GRID_COLS+1, UNKNOWN));

    out.interiorCells.clear();
    out.surfaceEdges.clear();
    out.bfsGraph.clear();

    // --- Optimized Pre‑compute density grid ---
    vector<vector<float>> densityGrid(GRID_ROWS+1, vector<float>(GRID_COLS+1, 1.0f));
    int cellArea = CELL_W * CELL_H;
    auto computeDensity = [&](int cx, int cy) -> float {
        int px = cx * CELL_W;
        int py = cy * CELL_H;
        if (px < 0 || py < 0 || px + CELL_W > FRAME_WIDTH || py + CELL_H > FRAME_HEIGHT) return 1.0f;
        if (!roi.contains(Point(px + CELL_W/2, py + CELL_H/2))) return 1.0f;
        Mat cell = barrierMap(Rect(px, py, CELL_W, CELL_H));
        int barrierPixels = countNonZero(cell);
        return (float)barrierPixels / (float)cellArea;
    };
    for (int cy = 0; cy <= GRID_ROWS; ++cy)
        for (int cx = 0; cx <= GRID_COLS; ++cx)
            densityGrid[cy][cx] = computeDensity(cx, cy);

    // --- Seed deployment ---
    int sx = std::min(std::max((int)(predictedSeed.x / CELL_W), 0), GRID_COLS);
    int sy = std::min(std::max((int)(predictedSeed.y / CELL_H), 0), GRID_ROWS);

    queue<Point> q;
    int interiorCount = 0;

    int offsetCells = 20 / CELL_W;
    int offsetsX[] = {0, offsetCells, -offsetCells, 0, 0};
    int offsetsY[] = {0, 0, 0, offsetCells, -offsetCells};
    bool seedDeployed = false;

    for(int i = 0; i < 5; i++) {
        int nx = sx + offsetsX[i];
        int ny = sy + offsetsY[i];
        if (nx >= 0 && nx <= GRID_COLS && ny >= 0 && ny <= GRID_ROWS) {
            if (vState[ny][nx] == UNKNOWN && densityGrid[ny][nx] < 0.25f) {
                vState[ny][nx] = INTERIOR;
                q.push({nx, ny});
                out.interiorCells.emplace_back(std::min(nx*CELL_W, FRAME_WIDTH-1), std::min(ny*CELL_H, FRAME_HEIGHT-1));
                interiorCount++;
                seedDeployed = true;
            }
        }
    }

    if (!seedDeployed) return out;

    const int ddx[4]={1,-1,0,0}, ddy[4]={0,0,1,-1};

    while (!q.empty()) {
        Point v = q.front(); q.pop();
        Point pParent(v.x * CELL_W + CELL_W/2, v.y * CELL_H + CELL_H/2);

        for (int i=0; i<4; i++) {
            int nx = v.x+ddx[i], ny = v.y+ddy[i];
            if (nx<0 || nx>GRID_COLS || ny<0 || ny>GRID_ROWS) continue;
            if (vState[ny][nx] == INTERIOR) continue;

            float density = densityGrid[ny][nx]; 
            Point pChild(nx * CELL_W + CELL_W/2, ny * CELL_H + CELL_H/2);

            if (density >= 0.25f) {
                vState[ny][nx] = SURFACE_CANDIDATE;
                out.surfaceEdges.push_back(pChild);
            } else {
                vState[ny][nx] = INTERIOR;
                out.bfsGraph.push_back({pParent, pChild});
                q.push({nx,ny});
                out.interiorCells.push_back(pChild);
                interiorCount++;
                if (interiorCount > maxCells) { out.success = false; return out; }
            }
        }
    }

    Mat interiorMask = Mat::zeros(GRID_ROWS+1, GRID_COLS+1, CV_8UC1);
    for (int r=0; r<=GRID_ROWS; r++)
        for (int c=0; c<=GRID_COLS; c++)
            if (vState[r][c] == INTERIOR) interiorMask.at<uchar>(r,c) = 255;

    out.interiorMask = interiorMask;
    out.success = true;
    return out;
}