#!/bin/bash
# Emotion Detection Project - Unix/Linux/macOS Script
# This script helps you run the emotion detection system easily

echo
echo "========================================"
echo " 🎭 EMOTION DETECTION FROM TEXT"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python is not installed or not in PATH"
        echo "Please install Python 3.8+ from https://python.org"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python is installed"
echo

# Show menu options
echo "Choose an option:"
echo
echo "1. 🚀 Quick Start (Run Web App)"
echo "2. 📦 Install Dependencies"
echo "3. 🤖 Train Model"
echo "4. 📊 Evaluate Model"
echo "5. 🌐 Run Web Application"
echo "6. 🧪 Run All Tests"
echo "7. ❓ Show Help"
echo "0. ❌ Exit"
echo

read -p "Enter your choice (0-7): " choice

case $choice in
    1)
        echo
        echo "🚀 Starting Quick Setup and Web App..."
        echo
        echo "Installing dependencies..."
        pip install -r requirements.txt
        echo
        echo "Navigating to src directory..."
        cd src
        echo
        echo "Starting Streamlit web application..."
        streamlit run app.py
        ;;
    2)
        echo
        echo "📦 Installing Dependencies..."
        echo
        pip install -r requirements.txt
        echo
        echo "Running setup script..."
        $PYTHON_CMD setup.py
        echo
        echo "✅ Installation completed!"
        read -p "Press Enter to continue..."
        ;;
    3)
        echo
        echo "🤖 Training Emotion Detection Model..."
        echo
        cd src
        $PYTHON_CMD train_model.py
        echo
        echo "✅ Model training completed!"
        read -p "Press Enter to continue..."
        ;;
    4)
        echo
        echo "📊 Evaluating Model Performance..."
        echo
        cd src
        $PYTHON_CMD evaluate_model.py
        echo
        echo "✅ Evaluation completed!"
        read -p "Press Enter to continue..."
        ;;
    5)
        echo
        echo "🌐 Starting Web Application..."
        echo
        cd src
        streamlit run app.py
        ;;
    6)
        echo
        echo "🧪 Running Complete Test Pipeline..."
        echo
        echo "Step 1: Installing dependencies..."
        pip install -r requirements.txt
        echo
        echo "Step 2: Training model..."
        cd src
        $PYTHON_CMD train_model.py
        echo
        echo "Step 3: Evaluating model..."
        $PYTHON_CMD evaluate_model.py
        echo
        echo "Step 4: Starting web app..."
        streamlit run app.py
        ;;
    7)
        echo
        echo "❓ HELP - Emotion Detection Project"
        echo "==================================="
        echo
        echo "This project detects emotions from text using Machine Learning."
        echo
        echo "📁 Project Structure:"
        echo "  src/           - Source code files"
        echo "  models/        - Trained model files"
        echo "  data/          - Dataset directory"
        echo "  requirements.txt - Python dependencies"
        echo
        echo "🔧 Main Components:"
        echo "  app.py           - Streamlit web application"
        echo "  train_model.py   - Model training script"
        echo "  evaluate_model.py - Model evaluation"
        echo "  preprocessing.py - Text preprocessing"
        echo "  utils.py         - Helper utilities"
        echo
        echo "🚀 Quick Start:"
        echo "  1. Run option 1 for automatic setup"
        echo "  2. Or install dependencies first (option 2)"
        echo "  3. Then run web app (option 5)"
        echo
        echo "🌐 Web App URL: http://localhost:8501"
        echo
        echo "📧 For issues: Check README.md or create GitHub issue"
        echo
        read -p "Press Enter to continue..."
        ;;
    0)
        echo
        echo "👋 Goodbye! Thanks for using Emotion Detection!"
        echo
        exit 0
        ;;
    *)
        echo
        echo "❌ Invalid choice. Please enter a number between 0-7."
        echo
        read -p "Press Enter to continue..."
        ;;
esac

echo