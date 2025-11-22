@echo off
REM Emotion Detection System - Menu Launcher
REM This script provides easy access to all features

setlocal enabledelayedexpansion
cd /d "%~dp0"

:menu
cls
echo.
echo ========================================
echo   EMOTION DETECTION SYSTEM
echo ========================================
echo.
echo Choose an option:
echo.
echo 1) Start Web UI (Gradio) - RECOMMENDED
echo 2) Webcam Detection (Real-time)
echo 3) View Quick Start Guide
echo 4) Process Image File
echo 5) Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto ui
if "%choice%"=="2" goto webcam
if "%choice%"=="3" goto guide
if "%choice%"=="4" goto image
if "%choice%"=="5" goto end
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:ui
echo.
echo Starting Gradio Web UI...
echo.
python simple_ui.py
goto menu

:webcam
echo.
echo Starting Webcam Detection...
echo Press 'q' to quit
echo.
python main.py --source 0 --show-fps
pause
goto menu

:guide
echo.
echo Showing Quick Start Guide...
echo.
python QUICK_START.py
goto menu

:image
echo.
set /p imgpath="Enter image path: "
set /p outpath="Enter output path (optional): "

if "!outpath!"=="" (
    echo Processing image...
    python main.py --source "!imgpath!"
) else (
    echo Processing and saving to !outpath!...
    python main.py --source "!imgpath!" --output-path "!outpath!"
)
pause
goto menu

:end
echo.
echo Goodbye!
echo.
