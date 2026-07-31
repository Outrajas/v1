// main.cpp – Targeted Elastic IK & Dimensional Orientation (Centroid Stabilized)
#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// ── Local compile‑time constants ──────────────────────────────────────────────
constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;
constexpr int GRID_COLS = 160;
constexpr int GRID_ROWS = 120;
constexpr int CELL_W = FRAME_WIDTH / GRID_COLS;
constexpr int CELL_H = FRAME_HEIGHT / GRID_ROWS;
constexpr float DISPLAY_SCALE = 0.65f;
constexpr int BILATERAL_D = 3;
constexpr double BILATERAL_SIG_COLOR = 25.0;
constexpr double BILATERAL_SIG_SPACE = 25.0;
constexpr float MAX_ROI_AREA_RATIO = 0.4f;
#define PI 3.14159265f

// ── Tracking Structs ──────────────────────────────────────────────────────────
struct TrackedObject {
    Mat prevInteriorMask;
    vector<vector<Point>> prevContours;
    Point2f centroid;
    float area;
    bool valid = false;
};
struct RayResult { Point2f start; Point2f end; bool hitBarrier; float support; };
struct EnclosureResult {
    float enclosureScore; float supportScore; float meshContinuity; float confidence;
    vector<RayResult> rays; vector<Point2f> anchors;
};
struct TopologyOutput {
    Mat interiorMask; vector<Point> surfaceEdges; vector<Point> interiorCells;
    vector<pair<Point, Point>> bfsGraph; bool success = false;
};

// ── Autonomous UI State Variables ─────────────────────────────────────────────
int ui_pos_x = 320;   
int ui_pos_y = 320;
int ui_pos_z = 500;  
int ui_pitch = 180; 
int ui_yaw   = 180;
int ui_roll  = 180;
int ui_splay = 15;
int ui_thumb_bend  = 0;
int ui_index_bend  = 0;
int ui_middle_bend = 0;
int ui_ring_bend   = 0;
int ui_pinky_bend  = 0;

// ── 3D Mathematics Helpers ────────────────────────────────────────────────────
Point3f rotate3D(Point3f pt, float pitch, float yaw, float roll) {
    float y1 = pt.y * cos(pitch) - pt.z * sin(pitch);
    float z1 = pt.y * sin(pitch) + pt.z * cos(pitch);
    float x2 = pt.x * cos(yaw) + z1 * sin(yaw);
    float z2 = -pt.x * sin(yaw) + z1 * cos(yaw);
    float x3 = x2 * cos(roll) - y1 * sin(roll);
    float y3 = x2 * sin(roll) + y1 * cos(roll);
    return Point3f(x3, y3, z2);
}

Point2f projectTo2D(Point3f pt, float offsetX, float offsetY, float offsetZ) {
    float focalLength = 600.0f;
    float z_cam = pt.z + offsetZ; 
    float scale = (z_cam > 0.1f) ? (focalLength / z_cam) : 0.01f;
    return Point2f((pt.x * scale) + offsetX, (pt.y * scale) + offsetY);
}

// ── 21-Point Kinematic Structures ─────────────────────────────────────────────
struct Finger3D {
    Point3f rootOffset; float lengths[3]; float baseSplay;        
    float maxBend; float currentBend; bool isThumb;
};

struct StandardHand21 {
    Point3f nodes3D[21]; Point2f nodes2D[21]; Finger3D fingers[5];    
};

void buildStandardHand(StandardHand21& hand) {
    hand.fingers[0].rootOffset = Point3f(-40.0f, -30.0f, 0.0f); 
    hand.fingers[1].rootOffset = Point3f(-25.0f, -95.0f, 0.0f);
    hand.fingers[2].rootOffset = Point3f(  0.0f, -100.0f, 0.0f);
    hand.fingers[3].rootOffset = Point3f( 25.0f, -95.0f, 0.0f);
    hand.fingers[4].rootOffset = Point3f( 45.0f, -80.0f, 0.0f);

    hand.fingers[0].isThumb = true; hand.fingers[0].maxBend = 90.0f;
    hand.fingers[0].lengths[0] = 40.0f; hand.fingers[0].lengths[1] = 30.0f; hand.fingers[0].lengths[2] = 20.0f; 

    float base_lengths[4] = {45.0f, 50.0f, 45.0f, 35.0f}; 
    for(int i = 1; i <= 4; i++) {
        hand.fingers[i].isThumb = false; hand.fingers[i].maxBend = 160.0f;
        hand.fingers[i].lengths[0] = base_lengths[i-1];             
        hand.fingers[i].lengths[1] = base_lengths[i-1] * 0.6f;      
        hand.fingers[i].lengths[2] = base_lengths[i-1] * 0.45f;     
    }
}

void solveKinematics(StandardHand21& hand) {
    float gPitch = (ui_pitch - 180) * (PI / 180.0f);
    float gYaw   = (ui_yaw - 180)   * (PI / 180.0f);
    float gRoll  = (ui_roll - 180)  * (PI / 180.0f);
    float splay  = ui_splay         * (PI / 180.0f);
    float offsetX = (float)ui_pos_x; float offsetY = (float)ui_pos_y; float offsetZ = (float)ui_pos_z;

    hand.fingers[0].currentBend = ui_thumb_bend  / 100.0f;
    hand.fingers[1].currentBend = ui_index_bend  / 100.0f;
    hand.fingers[2].currentBend = ui_middle_bend / 100.0f;
    hand.fingers[3].currentBend = ui_ring_bend   / 100.0f;
    hand.fingers[4].currentBend = ui_pinky_bend  / 100.0f;

    hand.fingers[1].baseSplay = -splay * 1.5f;
    hand.fingers[2].baseSplay = -splay * 0.5f;
    hand.fingers[3].baseSplay =  splay * 0.5f;
    hand.fingers[4].baseSplay =  splay * 1.5f;

    Point3f localNodes[21]; localNodes[0] = Point3f(0, 0, 0); 

    for (int i = 0; i < 5; i++) {
        Finger3D& f = hand.fingers[i];
        int baseIdx = 1 + (i * 4); 
        Point3f mcpLocal = f.rootOffset;
        localNodes[baseIdx] = mcpLocal;
        
        float totalBendRad = f.currentBend * f.maxBend * (PI / 180.0f);
        float j1Flex = totalBendRad * 0.4f; float j2Flex = totalBendRad * 0.4f;

        if (f.isThumb) {
            Point3f dir(-0.7071f, -0.7071f, 0.0f); 
            Point3f seg1 = rotate3D(dir, -j1Flex, j1Flex * 1.2f, 0);
            localNodes[baseIdx+1] = mcpLocal + Point3f(seg1.x * f.lengths[0], seg1.y * f.lengths[0], seg1.z * f.lengths[0]);
            Point3f seg2 = rotate3D(dir, -(j1Flex + j2Flex), (j1Flex + j2Flex) * 1.2f, 0);
            localNodes[baseIdx+2] = localNodes[baseIdx+1] + Point3f(seg2.x * f.lengths[1], seg2.y * f.lengths[1], seg2.z * f.lengths[1]);
            Point3f seg3 = rotate3D(dir, -totalBendRad, totalBendRad * 1.2f, 0);
            localNodes[baseIdx+3] = localNodes[baseIdx+2] + Point3f(seg3.x * f.lengths[2], seg3.y * f.lengths[2], seg3.z * f.lengths[2]);
        } else {
            Point3f dir1 = rotate3D(Point3f(0, -f.lengths[0], 0), -j1Flex, 0, f.baseSplay);
            localNodes[baseIdx+1] = mcpLocal + dir1;
            Point3f dir2 = rotate3D(Point3f(0, -f.lengths[1], 0), -(j1Flex + j2Flex), 0, f.baseSplay);
            localNodes[baseIdx+2] = localNodes[baseIdx+1] + dir2;
            Point3f dir3 = rotate3D(Point3f(0, -f.lengths[2], 0), -totalBendRad, 0, f.baseSplay);
            localNodes[baseIdx+3] = localNodes[baseIdx+2] + dir3;
        }
    }

    Point3f palmCentroid = (localNodes[0] + localNodes[1] + localNodes[5] + localNodes[9] + localNodes[13] + localNodes[17]);
    palmCentroid.x /= 6.0f; palmCentroid.y /= 6.0f; palmCentroid.z /= 6.0f;

    for (int i = 0; i < 21; i++) {
        Point3f centeredNode = localNodes[i] - palmCentroid;
        hand.nodes3D[i] = rotate3D(centeredNode, gPitch, gYaw, gRoll);
        hand.nodes2D[i] = projectTo2D(hand.nodes3D[i], offsetX, offsetY, offsetZ); 
    }
}

// ── True Targeted Elastic IK ──────────────────────────────────────────────────
void applyTargetedKinematics(const vector<Point>& contour, const Mat& interiorMask, StandardHand21& hand, Point2f centroid, float area) {
    if (contour.size() < 20 || area < 100) return;

    // 1. DIMENSIONAL SCALING (Z-Depth)
    float scale = sqrt(35000.0f / area); 
    ui_pos_z = std::max(100, std::min(2000, (int)(500.0f * scale)));

    // 2. TRUE ROLL INFERENCE (Aimed at the absolute furthest peak)
    Point2f farthest_pt = centroid;
    float max_dist = 0;
    for (const auto& pt : contour) {
        float d = norm(Point2f(pt) - centroid);
        if (d > max_dist) { max_dist = d; farthest_pt = Point2f(pt); }
    }

    static float smoothed_roll = 180.0f;
    float rollAngle = atan2(farthest_pt.y - centroid.y, farthest_pt.x - centroid.x);
    float target_roll = (rollAngle * 180.0f / PI) + 270.0f; 
    target_roll = fmod(target_roll, 360.0f);
    
    float diff = target_roll - smoothed_roll;
    if (diff > 180.0f) diff -= 360.0f;
    if (diff < -180.0f) diff += 360.0f;
    smoothed_roll += diff * 0.2f; 
    ui_roll = (int)smoothed_roll;

    // 3. YAW & PITCH INFERENCE (Foreshortening & Asymmetry)
    Rect bb = boundingRect(contour);
    Point2f bbCenter(bb.x + bb.width / 2.0f, bb.y + bb.height / 2.0f);
    float yawOffset = (centroid.x - bbCenter.x) / (bb.width / 2.0f);
    ui_yaw = 180 + (int)(yawOffset * -45.0f); 

    float expected_length = sqrt(area) * 1.5f;
    float pitchRatio = std::max(0.1f, std::min(1.0f, max_dist / expected_length));
    ui_pitch = 180 + (int)((1.0f - pitchRatio) * 60.0f);

    setTrackbarPos("Pos Z (Depth)", "Controls", ui_pos_z);
    setTrackbarPos("Roll (Tilt)", "Controls", ui_roll);
    setTrackbarPos("Yaw (Turn)", "Controls", ui_yaw);
    setTrackbarPos("Pitch (Fold)", "Controls", ui_pitch);

    solveKinematics(hand);

    // 4. ELASTIC TARGET PEAK EXTRACTION
    static vector<int> hullIndices; hullIndices.clear(); hullIndices.reserve(100);
    convexHull(contour, hullIndices, false, false);
    
    static vector<Point2f> cleanPeaks; cleanPeaks.clear(); cleanPeaks.reserve(30);
    for (int idx : hullIndices) {
        Point2f p = Point2f(contour[idx]);
        bool duplicate = false;
        for (const auto& cp : cleanPeaks) {
            if (norm(p - cp) < 25.0f) { duplicate = true; break; }
        }
        if (!duplicate) cleanPeaks.push_back(p);
    }

    // 5. THE GULLEY FITTER (O(1) Boundary Safe Lookup)
    int bestBends[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < 5; i++) {
        int tipIdx = (i * 4) + 4;
        
        if(i==0) ui_thumb_bend = 0;
        if(i==1) ui_index_bend = 0;
        if(i==2) ui_middle_bend = 0;
        if(i==3) ui_ring_bend = 0;
        if(i==4) ui_pinky_bend = 0;
        solveKinematics(hand);
        Point2f extendedPos = hand.nodes2D[tipIdx];

        Point2f targetPeak = extendedPos;
        float minTargetDist = 1e6;
        for (const auto& pk : cleanPeaks) {
            float d = norm(extendedPos - pk);
            if (d < minTargetDist && d < 120.0f) { 
                minTargetDist = d;
                targetPeak = pk;
            }
        }

        int bestBend = 0;
        float minBendDist = 1e6;
        
        for (int testBend = 0; testBend <= 100; testBend += 5) {
            if(i==0) ui_thumb_bend = testBend;
            if(i==1) ui_index_bend = testBend;
            if(i==2) ui_middle_bend = testBend;
            if(i==3) ui_ring_bend = testBend;
            if(i==4) ui_pinky_bend = testBend;
            solveKinematics(hand);

            float distToPeak = norm(hand.nodes2D[tipIdx] - targetPeak);
            
            int gridX = cvRound(hand.nodes2D[tipIdx].x) / CELL_W;
            int gridY = cvRound(hand.nodes2D[tipIdx].y) / CELL_H;
            
            if (gridX >= 0 && gridX <= GRID_COLS && gridY >= 0 && gridY <= GRID_ROWS) {
                if (interiorMask.at<uchar>(gridY, gridX) == 0) {
                    distToPeak += 500.0f; // Penalize out of bounds
                }
            } else {
                distToPeak += 500.0f; 
            }

            if (distToPeak < minBendDist) {
                minBendDist = distToPeak;
                bestBend = testBend;
            }
        }
        bestBends[i] = bestBend;
    }

    ui_thumb_bend  = bestBends[0]; ui_index_bend  = bestBends[1];
    ui_middle_bend = bestBends[2]; ui_ring_bend   = bestBends[3];
    ui_pinky_bend  = bestBends[4];

    setTrackbarPos("Thumb Bend",  "Controls", ui_thumb_bend);
    setTrackbarPos("Index Bend",  "Controls", ui_index_bend);
    setTrackbarPos("Middle Bend", "Controls", ui_middle_bend);
    setTrackbarPos("Ring Bend",   "Controls", ui_ring_bend);
    setTrackbarPos("Pinky Bend",  "Controls", ui_pinky_bend);

    solveKinematics(hand); 
}

void renderSimulation(Mat& canvas, const StandardHand21& hand) {
    Scalar boneColor(255, 150, 0); 
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            int nodeStart = (i * 4) + 1 + j; int nodeEnd = nodeStart + 1;
            line(canvas, hand.nodes2D[nodeStart], hand.nodes2D[nodeEnd], boneColor, 2, LINE_AA);
        }
        line(canvas, hand.nodes2D[0], hand.nodes2D[(i * 4) + 1], boneColor, 2, LINE_AA);
    }
    line(canvas, hand.nodes2D[1], hand.nodes2D[5], boneColor, 2, LINE_AA); 
    line(canvas, hand.nodes2D[5], hand.nodes2D[9], boneColor, 2, LINE_AA);
    line(canvas, hand.nodes2D[9], hand.nodes2D[13], boneColor, 2, LINE_AA);
    line(canvas, hand.nodes2D[13], hand.nodes2D[17], boneColor, 2, LINE_AA);

    for (int i = 0; i < 21; i++) {
        Scalar nodeColor = (i == 0) ? Scalar(0, 0, 255) : 
                           (i % 4 == 0 && i != 0) ? Scalar(0, 255, 0) : Scalar(0, 255, 255); 
        float pseudoDepth = (ui_pos_z + hand.nodes3D[i].z);
        int dotSize = std::max(2, (int)(2500.0f / (pseudoDepth + 100.0f))); 
        circle(canvas, hand.nodes2D[i], dotSize, nodeColor, -1, LINE_AA); 
    }
}

// ── External Global State ─────────────────────────────────────────────────────
extern Mat globalReferenceVisual;
extern Mat globalStructuralBarrier;
extern Mat globalRawStacked;
extern Rect currentROI;
extern Point2f current_seed;
extern Point2f roiCenter;
extern Point2f rawMeasuredCentroid;
extern bool trackingRobust;
extern int lostFrames;
extern int fps, frameCount;
extern chrono::steady_clock::time_point lastTime;
extern TrackedObject trackedObject;
extern TopologyOutput lastTopology;
extern EnclosureResult lastEnclosure;
extern float gLeakConfidence;

extern void extractEdges(const Mat& frame, const Rect& roi, Mat& refVisualOut, Mat& structuralBarrierOut);
extern TopologyOutput runTopologyBFS(const Mat& barrierMap, const Rect& roi, Point2f predictedSeed, const Mat& prevMask, bool havePrev, float leakConf);
extern EnclosureResult analyzeEnclosure(const vector<Point>& interiorCells, const Mat& barrierMap, const Rect& roi);
extern vector<Point> extractOuterContour(const Mat& interiorMask, const Mat& barrierMap);
extern void drawGrid(Mat& img);

int main() {
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cap.set(CAP_PROP_FPS, 60);
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    if (!cap.isOpened()) { cerr << "Cannot open camera\n"; return 1; }

    namedWindow("Stabilized Anchor Tracker", WINDOW_AUTOSIZE);
    namedWindow("Dimensional Kinematic Cage", WINDOW_AUTOSIZE);

    namedWindow("Controls", WINDOW_NORMAL);
    resizeWindow("Controls", 400, 450);
    
    createTrackbar("Pos Z (Depth)", "Controls", nullptr, 2000);
    createTrackbar("Pitch (Fold)",  "Controls", nullptr, 360);
    createTrackbar("Yaw (Turn)",    "Controls", nullptr, 360);
    createTrackbar("Roll (Tilt)",   "Controls", nullptr, 360);
    createTrackbar("Finger Splay",  "Controls", nullptr, 45);
    createTrackbar("Thumb Bend",    "Controls", nullptr, 100);
    createTrackbar("Index Bend",    "Controls", nullptr, 100);
    createTrackbar("Middle Bend",   "Controls", nullptr, 100);
    createTrackbar("Ring Bend",     "Controls", nullptr, 100);
    createTrackbar("Pinky Bend",    "Controls", nullptr, 100);

    Mat frame;
    globalReferenceVisual   = Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
    globalStructuralBarrier = Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
    globalRawStacked        = Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);

    StandardHand21 augmentedHand;
    buildStandardHand(augmentedHand);

    static Mat sharedCanvas(FRAME_HEIGHT * 2, FRAME_WIDTH * 2, CV_8UC3, Scalar(0,0,0));
    Mat q1 = sharedCanvas(Rect(0, 0, FRAME_WIDTH, FRAME_HEIGHT));
    Mat q2 = sharedCanvas(Rect(FRAME_WIDTH, 0, FRAME_WIDTH, FRAME_HEIGHT));
    Mat q3 = sharedCanvas(Rect(0, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT));
    Mat q4 = sharedCanvas(Rect(FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT));
    
    static Mat smooth, roiSmooth, display, cageCanvas;
    static Mat coloredBarrier, coloredRaw; 

    lastTime = chrono::steady_clock::now();

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        flip(frame, frame, 1);
        frameCount++;

        Rect roi = currentROI & Rect(0,0,FRAME_WIDTH,FRAME_HEIGHT);
        
        frame.copyTo(smooth);
        if (roi.area() > 0) {
            bilateralFilter(frame(roi), roiSmooth, BILATERAL_D, BILATERAL_SIG_COLOR, BILATERAL_SIG_SPACE);
            roiSmooth.copyTo(smooth(roi));
        }

        extractEdges(smooth, roi, globalReferenceVisual, globalStructuralBarrier);

        Point2f predictedCenter = (trackedObject.valid && !trackedObject.prevContours.empty()) ? trackedObject.centroid : current_seed;
        bool havePrev = trackedObject.valid && !trackedObject.prevInteriorMask.empty();
        
        lastTopology = runTopologyBFS(globalStructuralBarrier, roi, predictedCenter, trackedObject.prevInteriorMask, havePrev, gLeakConfidence);

        if (lastTopology.success) {
            lastEnclosure = analyzeEnclosure(lastTopology.interiorCells, globalStructuralBarrier, roi);
            gLeakConfidence = lastEnclosure.confidence;
        } else {
            lastEnclosure = EnclosureResult();
            gLeakConfidence = 0.5f;
        }

        bool objectUpdated = false;
        float newArea = 0;
        vector<Point> finalContour;

        if (lastTopology.success) {
            finalContour = extractOuterContour(lastTopology.interiorMask, globalStructuralBarrier);
            if (!finalContour.empty()) {
                // THE FIX: Because finalContour is now an open 1D raw line, calculating moments 
                // directly on it yields a chaotic near-zero area, causing the centroid to teleport.
                // We use the Convex Hull to establish a perfectly stable mathematical footprint.
                vector<Point> spatialHull;
                convexHull(finalContour, spatialHull);
                Moments mu = moments(spatialHull);
                
                if (mu.m00 > 100) { 
                    rawMeasuredCentroid = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
                    newArea = mu.m00;
                    objectUpdated = true;

                    ui_pos_x = cvRound(rawMeasuredCentroid.x);
                    ui_pos_y = cvRound(rawMeasuredCentroid.y);
                    
                    applyTargetedKinematics(finalContour, lastTopology.interiorMask, augmentedHand, rawMeasuredCentroid, newArea);
                }
            }
        }

        if (objectUpdated) {
            current_seed = rawMeasuredCentroid;
            Rect bbox = boundingRect(finalContour);
            int margin = 10;
            bbox.x -= margin; bbox.y -= margin;
            bbox.width  += 2 * margin; bbox.height += 2 * margin;
            bbox = bbox & Rect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);

            const int totalArea = FRAME_WIDTH * FRAME_HEIGHT;
            const int maxArea = static_cast<int>(totalArea * MAX_ROI_AREA_RATIO);
            if (bbox.width * bbox.height > maxArea) {
                float scale = sqrt(static_cast<float>(maxArea) / (bbox.width * bbox.height));
                int new_w = cvRound(bbox.width * scale);
                int new_h = cvRound(bbox.height * scale);
                int new_x = cvRound(rawMeasuredCentroid.x - new_w / 2.0f);
                int new_y = cvRound(rawMeasuredCentroid.y - new_h / 2.0f);
                bbox = Rect(new_x, new_y, new_w, new_h) & Rect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
                if (bbox.width <= 0) bbox.width = 1;
                if (bbox.height <= 0) bbox.height = 1;
            }

            currentROI = bbox;
            roiCenter = rawMeasuredCentroid;

            lastTopology.interiorMask.copyTo(trackedObject.prevInteriorMask);
            trackedObject.centroid = current_seed;
            trackedObject.area = newArea;
            trackedObject.valid = true;
            trackingRobust = true;
            lostFrames = 0;
            trackedObject.prevContours = {finalContour};
        } else {
            lostFrames++;
            if (lostFrames > 10) {
                trackedObject.valid = false;
                trackingRobust = false;
            } else {
                trackingRobust = true;
            }
            solveKinematics(augmentedHand); 
        }

        // ====================================================================
        // VIEW 1: Direct Memory Render (Zero allocations)
        // ====================================================================
        cvtColor(globalStructuralBarrier, coloredBarrier, COLOR_GRAY2BGR);
        coloredBarrier.copyTo(q1); 
        rectangle(q1, currentROI, Scalar(0,255,255), 1);
        putText(q1, "CONSENSUS MASK", Point(10,25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);

        cvtColor(globalRawStacked, coloredRaw, COLOR_GRAY2BGR);
        coloredRaw.copyTo(q2); 
        rectangle(q2, currentROI, Scalar(0,255,255), 1);
        putText(q2, "RAW STACKED SIGNAL", Point(10,25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);

        q3.setTo(Scalar(0,0,0));
        drawGrid(q3);
        addWeighted(q3, 1.0, coloredBarrier, 0.3, 0, q3); 

        if (lastTopology.success && !lastTopology.bfsGraph.empty()) {
            for (const auto& edge : lastTopology.bfsGraph) {
                line(q3, edge.first, edge.second, Scalar(150, 100, 50), 1);
            }
            for (const auto& p : lastTopology.surfaceEdges) {
                circle(q3, p, 2, Scalar(0, 255, 0), -1);
            }
        }
        
        circle(q3, rawMeasuredCentroid, 5, Scalar(0,0,255), 2);
        circle(q3, predictedCenter, 5, Scalar(0,255,255), 2);
        putText(q3, "TOPOLOGY + RAYS", Point(10,65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);

        frame.copyTo(q4);
        if (!trackedObject.prevContours.empty()) {
            drawContours(q4, trackedObject.prevContours, -1, Scalar(0,255,255), 2);
        }
        rectangle(q4, currentROI, Scalar(0,255,0), 2);
        putText(q4, "FINAL CONTOUR", Point(10,25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,0,255), 2);

        auto now = chrono::steady_clock::now();
        if (chrono::duration<double>(now - lastTime).count() >= 1.0) { fps = frameCount; frameCount = 0; lastTime = now; }
        rectangle(sharedCanvas, Rect(sharedCanvas.cols/2-100,0,200,40), Scalar(0,0,0), FILLED);
        putText(sharedCanvas, "FPS: "+to_string(fps), Point(sharedCanvas.cols/2-80,28), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);

        resize(sharedCanvas, display, Size(), DISPLAY_SCALE, DISPLAY_SCALE, INTER_LINEAR);
        imshow("Stabilized Anchor Tracker", display);

        // ====================================================================
        // VIEW 2: The Kinematic Cage 
        // ====================================================================
        if (cageCanvas.empty()) cageCanvas.create(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
        cageCanvas.setTo(Scalar(0,0,0));
        
        if (trackedObject.valid && !trackedObject.prevContours.empty()) {
            // Draw the pure 1D edge trace
            drawContours(cageCanvas, trackedObject.prevContours, -1, Scalar(255, 100, 100), 2);
            renderSimulation(cageCanvas, augmentedHand);
        }
        putText(cageCanvas, "TARGETED ELASTIC IK", Point(10,30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255), 2);
        imshow("Dimensional Kinematic Cage", cageCanvas);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}