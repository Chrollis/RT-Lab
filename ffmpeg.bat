@echo off
setlocal enabledelayedexpansion

:: Save the original directory (where the script is run from)
set "original_dir=%cd%"

echo ==============================================
echo   Convert BMP Sequence (frame_001~360) to Video
echo ==============================================
echo.

:: 1. Input image folder path (default: current directory)
set /p folder=Enter image folder path (or press Enter for current dir): 
if "%folder%"=="" set folder=.

:: 2. Input frame rate (default: 24)
set /p fps=Enter frame rate (default 24): 
if "%fps%"=="" set fps=24

:: 3. Input CRF quality (default: 18, lower = better)
set /p crf=Enter CRF value (default 18, 0~51, lower is better): 
if "%crf%"=="" set crf=18

:: 4. Input output file name (default: output.mp4)
set /p output=Enter output file name (default output.mp4): 
if "%output%"=="" set output=output.mp4

:: 5. Change to the specified image folder
cd /d "%folder%" 2>nul
if errorlevel 1 (
    echo [ERROR] Invalid folder path: "%folder%"
    pause
    exit /b 1
)

:: 6. Show settings and start conversion
echo.
echo Settings:
echo   Image folder = %cd%
echo   FPS          = %fps%
echo   CRF          = %crf%
echo   Output file  = %original_dir%\%output%
echo.
echo Starting conversion...

:: Run ffmpeg, output to the original directory (not the image folder)
ffmpeg -framerate %fps% -i frame_%%03d.bmp -c:v libx264 -crf %crf% -pix_fmt yuv420p -y "%original_dir%\%output%"

if errorlevel 1 (
    echo.
    echo [ERROR] Conversion failed. Please check:
    echo   - FFmpeg is installed and in PATH
    echo   - Image files exist as frame_001.bmp ... frame_360.bmp
    echo   - The folder path is correct
) else (
    echo.
    echo [SUCCESS] Video created at: %original_dir%\%output%
)

pause