# 🎭 EMOTION DETECTION PROJECT - COMMAND REFERENCE

## 🚀 QUICK START - Easy Menu

### Windows Users:
```bash
# Double-click this file OR run in terminal:
run_apps.bat
```

This will show you a menu to choose which app to run!

---

## 📝 INDIVIDUAL COMMANDS

### Navigate to Project Directory First:
```bash
cd "C:\ML project lab\emotion-detection"
```

---

## 📝 TEXT EMOTION DETECTION

### Run Basic Text App:
```bash
cd src
streamlit run app.py
```
- Opens at: http://localhost:8501
- Press Ctrl+C to stop

### Run Enhanced Text App (Recommended):
```bash
cd src
streamlit run app_enhanced.py
```
- Opens at: http://localhost:8501
- Better UI and more samples
- Press Ctrl+C to stop

---

## 👤 FACE EMOTION DETECTION

### Option 1: Webcam App (Best Performance):
```bash
cd src
python webcam_emotion_detector.py
```
**Controls:**
- `q` - Quit
- `s` - Save screenshot
- `r` - Reset statistics
- `h` - Toggle help

### Option 2: Web-based Real-time (Browser):
```bash
cd src
streamlit run face_emotion_app_fixed.py --server.port 8504
```
- Opens at: http://localhost:8504
- Allow camera permissions
- Press Ctrl+C to stop

### Option 3: Image Upload (Easiest):
```bash
cd src
streamlit run simple_emotion_app.py --server.port 8505
```
- Opens at: http://localhost:8505
- Upload photos or use camera
- Press Ctrl+C to stop

---

## 🔧 TRAINING MODELS

### Train Text Model (Basic):
```bash
cd src
python train_model.py
```

### Train Improved Text Model:
```bash
cd src
python train_improved_model.py
```

### Evaluate Model Performance:
```bash
cd src
python evaluate_model.py
```

---

## 🎯 RECOMMENDED WORKFLOW

### First Time Setup:
```bash
# 1. Navigate to project
cd "C:\ML project lab\emotion-detection"

# 2. Install dependencies (one time only)
pip install -r requirements.txt

# 3. Train improved model (one time only)
cd src
python train_improved_model.py
```

### Running Apps:
```bash
# Easy way - Use the menu:
run_apps.bat

# OR run specific app directly (see commands above)
```

---

## 🔥 MOST POPULAR COMMANDS

### For Text Emotion:
```bash
cd "C:\ML project lab\emotion-detection\src"
streamlit run app_enhanced.py
```

### For Face Emotion (Webcam):
```bash
cd "C:\ML project lab\emotion-detection\src"
python webcam_emotion_detector.py
```

### For Face Emotion (Upload Image):
```bash
cd "C:\ML project lab\emotion-detection\src"
streamlit run simple_emotion_app.py
```

---

## 🛑 STOPPING APPLICATIONS

### For Streamlit Apps:
- Press `Ctrl + C` in terminal
- OR close the terminal window

### For Webcam App:
- Press `q` key
- OR press `Ctrl + C` in terminal

---

## 🐛 TROUBLESHOOTING COMMANDS

### If port is busy:
```bash
# For Streamlit, use different port:
streamlit run app.py --server.port 8502
```

### Kill all Python processes (if needed):
```bash
taskkill /F /IM python.exe
```

### Reinstall packages:
```bash
pip install -r requirements.txt --force-reinstall
```

### Check Python version:
```bash
python --version
```

---

## 📱 ACCESS URLs

After running, open these URLs in your browser:

- **Text Emotion**: http://localhost:8501
- **Face Web App**: http://localhost:8504
- **Image Upload**: http://localhost:8505

---

## ✨ TIPS

1. **Run batch file** for easy menu selection
2. **Use webcam app** for best real-time performance
3. **Use image upload** if webcam issues
4. **Train improved model** for better text accuracy
5. **Good lighting** improves face detection

---

**Need Help?** Check README.md for detailed documentation!
