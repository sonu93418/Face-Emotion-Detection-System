@echo off
echo ====================================================
echo    🎭 EMOTION DETECTION PROJECT - LAUNCHER 🎭
echo ====================================================
echo.
echo Choose which application to run:
echo.
echo   1. 📝 Enhanced Text Emotion Detection (Accurate)
echo   2. 📷 Webcam Face Emotion Detection (Real-time)
echo.
echo   0. Exit
echo.
echo ====================================================
set /p choice="Enter your choice (0-2): "

if "%choice%"=="1" goto text_enhanced
if "%choice%"=="2" goto face_webcam
if "%choice%"=="0" goto end
goto invalid

:text_enhanced
echo.
echo ====================================================
echo 📝 Starting Enhanced Text Emotion Detection...
echo ====================================================
echo.
echo Opening in browser at http://localhost:8501
echo.
echo Features:
echo   - Analyze text and tweets for emotions
echo   - Multiple emotion categories
echo   - High accuracy with confidence scores
echo   - Beautiful interactive interface
echo.
echo Press Ctrl+C to stop the application
echo ====================================================
cd src
streamlit run app_enhanced.py
goto end

:face_webcam
echo.
echo ====================================================
echo 📷 Starting Webcam Face Emotion Detection...
echo ====================================================
echo.
echo Features:
echo   - Real-time face detection
echo   - 7 emotion categories (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
echo   - Live statistics and confidence scores
echo   - High accuracy emotion recognition
echo.
echo CONTROLS:
echo   - Press 'q' to quit
echo   - Press 's' to save screenshot
echo   - Press 'r' to reset statistics
echo   - Press 'h' to toggle help display
echo.
echo ====================================================
pause
cd src
python webcam_emotion_detector.py
goto end

:invalid
echo.
echo ====================================================
echo ❌ Invalid choice! Please enter 1, 2, or 0.
echo ====================================================
pause
goto end

:end
echo.
echo ====================================================
echo Thank you for using Emotion Detection Project! 🎭
echo ====================================================
echo.
