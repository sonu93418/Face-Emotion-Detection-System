@echo off
REM Emotion Detection Project - Windows Batch Script
REM This script helps you run the emotion detection system easily

echo.
echo ========================================
echo  🎭 EMOTION DETECTION FROM TEXT
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed
echo.

REM Show menu options
echo Choose an option:
echo.
echo 1. 🚀 Quick Start (Run Web App)
echo 2. 📦 Install Dependencies 
echo 3. 🤖 Train Model
echo 4. 📊 Evaluate Model
echo 5. 🌐 Run Web Application
echo 6. 🧪 Run All Tests
echo 7. ❓ Show Help
echo 0. ❌ Exit
echo.

set /p choice="Enter your choice (0-7): "

if "%choice%"=="1" goto quickstart
if "%choice%"=="2" goto install
if "%choice%"=="3" goto train
if "%choice%"=="4" goto evaluate  
if "%choice%"=="5" goto webapp
if "%choice%"=="6" goto testall
if "%choice%"=="7" goto help
if "%choice%"=="0" goto exit
goto invalid

:quickstart
echo.
echo 🚀 Starting Quick Setup and Web App...
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Navigating to src directory...
cd src
echo.
echo Starting Streamlit web application...
streamlit run app.py
goto end

:install
echo.
echo 📦 Installing Dependencies...
echo.
pip install -r requirements.txt
echo.
echo Running setup script...
python setup.py
echo.
echo ✅ Installation completed!
pause
goto end

:train
echo.
echo 🤖 Training Emotion Detection Model...
echo.
cd src
python train_model.py
echo.
echo ✅ Model training completed!
pause
goto end

:evaluate
echo.
echo 📊 Evaluating Model Performance...
echo.
cd src
python evaluate_model.py
echo.
echo ✅ Evaluation completed!
pause
goto end

:webapp
echo.
echo 🌐 Starting Web Application...
echo.
cd src
streamlit run app.py
goto end

:testall
echo.
echo 🧪 Running Complete Test Pipeline...
echo.
echo Step 1: Installing dependencies...
pip install -r requirements.txt
echo.
echo Step 2: Training model...
cd src
python train_model.py
echo.
echo Step 3: Evaluating model...
python evaluate_model.py
echo.
echo Step 4: Starting web app...
streamlit run app.py
goto end

:help
echo.
echo ❓ HELP - Emotion Detection Project
echo ===================================
echo.
echo This project detects emotions from text using Machine Learning.
echo.
echo 📁 Project Structure:
echo   src/           - Source code files
echo   models/        - Trained model files
echo   data/          - Dataset directory
echo   requirements.txt - Python dependencies
echo.
echo 🔧 Main Components:
echo   app.py           - Streamlit web application
echo   train_model.py   - Model training script
echo   evaluate_model.py - Model evaluation
echo   preprocessing.py - Text preprocessing
echo   utils.py         - Helper utilities
echo.
echo 🚀 Quick Start:
echo   1. Run option 1 for automatic setup
echo   2. Or install dependencies first (option 2)
echo   3. Then run web app (option 5)
echo.
echo 🌐 Web App URL: http://localhost:8501
echo.
echo 📧 For issues: Check README.md or create GitHub issue
echo.
pause
goto end

:invalid
echo.
echo ❌ Invalid choice. Please enter a number between 0-7.
echo.
pause
goto end

:exit
echo.
echo 👋 Goodbye! Thanks for using Emotion Detection!
echo.
goto end

:end
echo.
pause