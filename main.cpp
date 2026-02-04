#include <opencv2/opencv.hpp>
#include <windows.h>
#include <iostream>
#include <atomic>
#include <chrono>
#include "MotionEstimator.h"
#include "SkinExtractor.h"
#include "PalmEstimator.h"
#include "FingerAnalyzer.h"
#include "TemporalSmoother.h"
#include "ManualCalibrator.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;
constexpr int FRAME_FPS = 60;

// ==================== GLOBAL STATE ====================
std::atomic<bool> programRunning(true);
std::atomic<bool> cursorEnabled(false);
std::atomic<bool> isCalibrated(false);

CalibrationResult calibrationData;
GeometryUpdater geometryUpdater;
ShapeAnchoredTracker shapeTracker;

cv::Point2f lastValidPalm(-1, -1);
cv::Point2f lastValidThumbBase(-1, -1);
cv::Point2f lastValidPinkyBase(-1, -1);
int failureFrameCount = 0;

// ==================== HELPER FUNCTION DECLARATIONS ====================
void printStartupBanner();
cv::VideoCapture initializeCamera();
void configureCamera(cv::VideoCapture& camera);
void updateAndDisplayFPS(cv::Mat& frame, int& frameCounter, int& fps, 
                         std::chrono::steady_clock::time_point& lastFPSUpdate);
bool shouldExitProgram(int key);
void maintainFrameRate(std::chrono::steady_clock::time_point frameStart);

// ==================== CURSOR CONTROLLER ====================
class FastCursorController {
    int screenWidth, screenHeight;
    cv::Point2f lastCursorPos;
    
    float clampScreenCoordinate(float value, float max) const {
        return std::max(0.0f, std::min(value, max));
    }
    
    float calculateVelocityFactor(float deltaLength) const {
        return std::min(1.0f, deltaLength / 100.0f);
    }
    
public:
    FastCursorController() : lastCursorPos(-1, -1) {
        screenWidth = GetSystemMetrics(SM_CXSCREEN);
        screenHeight = GetSystemMetrics(SM_CYSCREEN);
    }

    void move(const cv::Point2f& camPos, const cv::Point2f& thumbTip, const cv::Point2f& indexTip) {
        if (!cursorEnabled) return;
        
        // Convert camera coordinates to screen space
        float screenX = (camPos.x / FRAME_WIDTH) * screenWidth;
        float screenY = (camPos.y / FRAME_HEIGHT) * screenHeight;
        
        screenX = clampScreenCoordinate(screenX, static_cast<float>(screenWidth - 1));
        screenY = clampScreenCoordinate(screenY, static_cast<float>(screenHeight - 1));
        
        // Apply adaptive smoothing based on movement velocity
        if (lastCursorPos.x >= 0) {
            cv::Point2f delta(screenX - lastCursorPos.x, screenY - lastCursorPos.y);
            float deltaLen = cv::norm(delta);
            
            if (deltaLen > 0) {
                if (deltaLen < 2.0f) {
                    // Minimal movement - maintain position
                    screenX = lastCursorPos.x;
                    screenY = lastCursorPos.y;
                } else {
                    // Velocity-based adaptive smoothing
                    float velocityFactor = calculateVelocityFactor(deltaLen);
                    float alpha = 0.3f + 0.5f * (1.0f - velocityFactor);
                    
                    screenX = lastCursorPos.x + delta.x * alpha;
                    screenY = lastCursorPos.y + delta.y * alpha;
                    
                    // Limit maximum movement per frame
                    float newDeltaLen = cv::norm(cv::Point2f(screenX, screenY) - lastCursorPos);
                    if (newDeltaLen > 50.0f) {
                        cv::Point2f direction = delta * (1.0f / deltaLen);
                        screenX = lastCursorPos.x + direction.x * 50.0f;
                        screenY = lastCursorPos.y + direction.y * 50.0f;
                    }
                }
            }
        }
        
        lastCursorPos = cv::Point2f(screenX, screenY);
        SetCursorPos(static_cast<int>(screenX), static_cast<int>(screenY));
    }
    
    void clearHistory() {
        lastCursorPos = cv::Point2f(-1, -1);
    }
};

// ==================== INPUT HANDLING ====================
class InputHandler {
    bool cKeyPressed = false;
    bool tKeyPressed = false;
    bool rKeyPressed = false;
    
public:
    void handleCalibrationInput(EnhancedManualCalibrator& calibrator, cv::Mat& frame, int key) {
        if ((GetAsyncKeyState('C') & 0x8000) || key == 'c' || key == 'C') {
            if (!cKeyPressed && !calibrator.isCalibrating()) {
                calibrator.startCalibration(frame);
                cKeyPressed = true;
            }
        } else {
            cKeyPressed = false;
        }
    }
    
    void handleCursorToggle(FastCursorController& cursor, int key) {
        if ((GetAsyncKeyState('T') & 0x8000) || key == 't' || key == 'T') {
            if (!tKeyPressed) {
                cursorEnabled = !cursorEnabled;
                cursor.clearHistory();
                std::cout << "Cursor " << (cursorEnabled ? "ENABLED (state-aware)" : "DISABLED") << std::endl;
                tKeyPressed = true;
            }
        } else {
            tKeyPressed = false;
        }
    }
    
    void handleResetInput(int key) {
        if ((GetAsyncKeyState('R') & 0x8000) || key == 'r' || key == 'R') {
            if (!rKeyPressed && isCalibrated) {
                performSystemReset();
                rKeyPressed = true;
            }
        } else {
            rKeyPressed = false;
        }
    }
    
private:
    void performSystemReset() {
        calibrationData = CalibrationResult();
        isCalibrated = false;
        geometryUpdater.reset();
        shapeTracker.reset();
        lastValidPalm = cv::Point2f(-1, -1);
        lastValidThumbBase = cv::Point2f(-1, -1);
        lastValidPinkyBase = cv::Point2f(-1, -1);
        failureFrameCount = 0;
        std::cout << "Calibration and tracking reset!" << std::endl;
    }
};

// ==================== FRAME PROCESSING ====================
class FrameProcessor {
    MotionEstimator motionEstimator;
    SkinExtractor skinExtractor;
    PalmEstimator palmEstimator;
    
public:
    void processFrame(cv::Mat& frame, FastCursorController& cursorController) {
        // ----- STEP 1: Motion Estimation -----
        cv::Mat motionMask;
        float deltaStrength = 0.0f;
        motionEstimator.getMotionMask(frame, motionMask, deltaStrength);
        
        // ----- STEP 2: Skin Extraction -----
        float hsvConfidence = 0.0f;
        cv::Mat skinMask = skinExtractor.detectSkin(
            frame, hsvConfidence, isCalibrated,
            calibrationData.calibratedHSVLower, calibrationData.calibratedHSVUpper,
            geometryUpdater.isPalmValid()
        );
        
        // ----- STEP 3: Palm Estimation -----
        auto palmResult = palmEstimator.detect(frame, motionMask, skinMask, hsvConfidence);
        
        // ----- STEP 4: Shape Anchoring -----
        if (isCalibrated && !shapeTracker.isAnchored() && palmResult.handDetected) {
            const HandGeometryState& state = geometryUpdater.getState();
            bool anchored = shapeTracker.anchorShape(
                palmResult.contour,
                state.palmCenter,
                state.thumbBase,
                state.pinkyBase,
                state.palmRadius,
                state.avgFingerDistance,
                state.smoothedHandScale
            );
            
            if (anchored) {
                std::cout << "Shape anchor established." << std::endl;
            }
        }
        
        // ----- STEP 5: Cursor Control -----
        if (cursorEnabled && palmResult.handDetected && geometryUpdater.isPalmValid()) {
            controlCursorBasedOnFingerState(cursorController);
        }
        
        // ----- STEP 6: Visualization -----
        palmEstimator.draw(frame, palmResult, skinMask, motionMask);
    }
    
private:
    void controlCursorBasedOnFingerState(FastCursorController& cursorController) {
        FingerState fingerState = geometryUpdater.getFingerState();
        
        if (fingerState == FINGER_STATE_FIST) {
            cv::Point2f palmPos = geometryUpdater.getPalmCenter();
            cursorController.move(palmPos, cv::Point2f(-1, -1), cv::Point2f(-1, -1));
        } else {
            cv::Point2f middleTip = geometryUpdater.getMiddleFingerTip();
            cv::Point2f thumbTip = geometryUpdater.getThumbTip();
            cv::Point2f indexTip = geometryUpdater.getIndexTip();
            
            if (middleTip.x >= 0) {
                cursorController.move(middleTip, thumbTip, indexTip);
            }
        }
    }
};

// ==================== HELPER FUNCTIONS IMPLEMENTATIONS ====================
void printStartupBanner() {
    std::cout << "Starting Hand Tracker with FINGER STATE AWARENESS" << std::endl;
    std::cout << "ENHANCEMENTS APPLIED:" << std::endl;
    std::cout << "1. FINGER STATE CLASSIFICATION: Fist/Partial/Open detection" << std::endl;
    std::cout << "2. NO FINGER HALLUCINATION: Clean fist handling with reset" << std::endl;
    std::cout << "3. PARTIAL HAND SUPPORT: Only detected finger count, no forced 5 IDs" << std::endl;
    std::cout << "4. GEOMETRY CONSISTENCY CHECKS: Suppress face/body blobs without HSV gating" << std::endl;
    std::cout << "5. ENHANCED CONTOUR ANCHORING: Distance-based projection with no stickiness" << std::endl;
    std::cout << "6. AGGRESSIVE ADAPTIVE SMOOTHING: Fast motion = 15% smoothing, Still = 70%" << std::endl;
    std::cout << "7. VELOCITY-ADAPTIVE CURSOR: Fast moves = responsive, slow moves = stable" << std::endl;
    std::cout << "\n=== STATE-AWARE HAND TRACKER ===" << std::endl;
    std::cout << "CORE IMPROVEMENTS:" << std::endl;
    std::cout << "1. Finger State Awareness: Knows when you make a fist" << std::endl;
    std::cout << "2. No Hallucination: Zero fingers = zero detected fingertips" << std::endl;
    std::cout << "3. Geometry Filtering: Face/body blobs suppressed by shape analysis" << std::endl;
    std::cout << "4. Responsive Fast Motion: 15% smoothing during rapid movement" << std::endl;
    std::cout << "5. Stable Rest: 70% smoothing when hand is still" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "C: Start calibration" << std::endl;
    std::cout << "T: Toggle cursor" << std::endl;
    std::cout << "R: Reset calibration" << std::endl;
    std::cout << "ESC: Exit" << std::endl;
    std::cout << "==========================================================" << std::endl;
}

cv::VideoCapture initializeCamera() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        cap.open(1);
    }
    return cap;
}

void configureCamera(cv::VideoCapture& camera) {
    camera.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    camera.set(cv::CAP_PROP_FPS, FRAME_FPS);
    std::cout << "Running at " << FRAME_WIDTH << "x" << FRAME_HEIGHT << " @ " << FRAME_FPS << " FPS" << std::endl;
}

void updateAndDisplayFPS(cv::Mat& frame, int& frameCounter, int& fps, 
                         std::chrono::steady_clock::time_point& lastFPSUpdate) {
    frameCounter++;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFPSUpdate);
    
    if (elapsed.count() >= 1000) {
        fps = frameCounter;
        frameCounter = 0;
        lastFPSUpdate = now;
    }
    
    std::string fpsText = "FPS: " + std::to_string(fps);
    cv::putText(frame, fpsText, cv::Point(FRAME_WIDTH - 80, FRAME_HEIGHT - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}

bool shouldExitProgram(int key) {
    return (key == 27 || key == 'q' || key == 'Q');
}

void maintainFrameRate(std::chrono::steady_clock::time_point frameStart) {
    auto frameEnd = std::chrono::steady_clock::now();
    auto frameTime = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
    
    int targetFrameTime = 1000 / FRAME_FPS;
    int remainingTime = std::max(1, targetFrameTime - static_cast<int>(frameTime));
    cv::waitKey(remainingTime);
}

// ==================== MAIN APPLICATION ====================
// ==================== MAIN APPLICATION ====================
int main() {
    // ----- INITIALIZATION -----
    printStartupBanner();
    
    cv::VideoCapture camera = initializeCamera();
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Cannot open camera!" << std::endl;
        return -1;
    }
    
    configureCamera(camera);
    
    FastCursorController cursor;
    InputHandler inputHandler;
    FrameProcessor frameProcessor;
    EnhancedManualCalibrator calibrator;
    
    cv::namedWindow("Hand Tracker - State Aware");
    
    // ----- MAIN LOOP -----
    cv::Mat frame;
    int frameCounter = 0;
    auto lastFPSUpdate = std::chrono::steady_clock::now();
    int fps = 0;
    
    while (programRunning) {
        auto frameStart = std::chrono::steady_clock::now();
        
        // Frame Acquisition
        if (!camera.read(frame) || frame.empty()) {
            std::cerr << "ERROR: Failed to read frame!" << std::endl;
            break;
        }
        
        cv::flip(frame, frame, 1);
        int key = cv::waitKey(1);
        
        // Input handling for calibration
        inputHandler.handleCalibrationInput(calibrator, frame, key);
        
        // Calibration Mode
        if (calibrator.isCalibrating()) {
            static bool mouseCallbackSet = false;
            if (!mouseCallbackSet) {
                cv::setMouseCallback("Hand Tracker - State Aware", [](int event, int x, int y, int flags, void* userdata) {
                    EnhancedManualCalibrator* cal = static_cast<EnhancedManualCalibrator*>(userdata);
                    cal->handleMouse(event, x, y);
                }, &calibrator);
                mouseCallbackSet = true;
            }
            
            cv::Mat displayFrame = frame.clone();  // Work on a copy
            
            if (calibrator.processCalibration(key, displayFrame)) {
                isCalibrated = true;
                geometryUpdater.reset();
                shapeTracker.reset();
                cv::setMouseCallback("Hand Tracker - State Aware", nullptr, nullptr);
                mouseCallbackSet = false;
                
                if (isCalibrated) {
                    std::cout << "Calibration complete. Enhanced tracking enabled." << std::endl;
                }
            }
            
            cv::imshow("Hand Tracker - State Aware", displayFrame);
            
            if (shouldExitProgram(key)) {
                programRunning = false;
                break;
            }
            continue;
        } else {
            cv::setMouseCallback("Hand Tracker - State Aware", nullptr, nullptr);
        }
        
        // ----- PIPELINE EXECUTION -----
        frameProcessor.processFrame(frame, cursor);
        
        // ----- INPUT HANDLING -----
        inputHandler.handleCursorToggle(cursor, key);
        inputHandler.handleResetInput(key);
        
        // ----- PERFORMANCE MONITORING -----
        updateAndDisplayFPS(frame, frameCounter, fps, lastFPSUpdate);
        
        // ----- RENDERING -----
        cv::imshow("Hand Tracker - State Aware", frame);
        
        // ----- EXIT CHECK -----
        if (shouldExitProgram(key)) {
            programRunning = false;
            break;
        }
        
        // ----- FRAME TIMING -----
        maintainFrameRate(frameStart);
    }
    
    // ----- CLEANUP -----
    camera.release();
    cv::destroyAllWindows();
    
    std::cout << "\nProgram terminated." << std::endl;
    return 0;
}