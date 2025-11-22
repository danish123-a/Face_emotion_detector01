@echo off
REM Emotion Detection - Webcam
REM Detects faces and emotions from webcam in real-time

echo.
echo ╔════════════════════════════════════════════╗
echo ║      EMOTION DETECTION - WEBCAM            ║
echo ╚════════════════════════════════════════════╝
echo.
echo Starting webcam emotion detection...
echo Press 'q' to quit
echo.

python main.py --source 0 --show-fps

pause
