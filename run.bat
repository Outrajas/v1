@echo off
setlocal

REM =======================================================
REM  Hand Reconstruction V1 – Standalone Compile & Run
REM =======================================================

cd /d "%~dp0" || exit /b 1

REM Load MSVC environment (Adjust path if needed for your system)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM OpenCV paths based on your previous configuration
set OPENCV_INCLUDE=C:\Users\ojasa\Downloads\opencv\build\include
set OPENCV_LIB=C:\Users\ojasa\Downloads\opencv\build\x64\vc16\lib

REM Compile ONLY the required files. explicitly avoiding main.cpp
cl /std:c++17 /EHsc ^
    /I"%OPENCV_INCLUDE%" ^
    /Fe:hand_reconstruction_v1.exe ^
    hand_reconstruction_v1.cpp globals.cpp edge_extraction.cpp topology.cpp enclosure.cpp ^
    /link opencv_world4120.lib user32.lib gdi32.lib ^
    /LIBPATH:"%OPENCV_LIB%"

if errorlevel 1 (
    echo.
    echo ❌ Compilation failed!
    pause
    exit /b 1
)

echo.
echo ✅ Compilation successful.
echo ▶ Launching Standalone Reconstruction Test...
echo.

REM Run the newly built standalone test
.\hand_reconstruction_v1.exe

echo.
echo Reconstruction Engine closed.
pause