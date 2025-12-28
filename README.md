# ASL Alphabet Recognition

Real-time ASL alphabet recognition using MediaPipe Hand Landmarks and an SVM classifier.

---

## 🚀 Quick Start (Web Application)

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

Requires **Python 3.8+**.

---

### **2. Run the Web Application**

**Method 1: Using the start script (Recommended)**
```bash
./run.sh
```

**Method 2: Manual Start**
```bash
cd frontend
../.venv/bin/python app.py
```
Wait until you see:
```
✅ Models loaded successfully!
 * Running on http://0.0.0.0:5001
```

Then open your browser and go to: **http://localhost:5001**

---

## 🎮 Web Application Features

* 🎥 **Live webcam detection** with MediaPipe hand landmarks (blue)
* 🔤 **Real-time ASL predictions** with confidence scores
* 📝 **Automatic sentence builder** - letters are added automatically
* 🎛️ **Interactive controls:**
  - **Add Space** - Add space to sentence
  - **Delete Last** - Remove last character
  - **Reset Sentence** - Clear entire sentence

---

## ⚙️ Advanced Options

### Testing via Console (For Debugging)

If you want to test directly without the web interface:

```bash
cd model
python -m src.inference.realtime
```

**Keyboard controls:**
- `Q` - Quit
- `SPACE` - Add space
- `BACKSPACE` - Delete last
- `R` - Reset

> ⚠️ **Note:** This is for testing/debugging only. For normal usage, use the web application above.

---

### Manual Model Training (Optional)

The model is already trained and ready to use. But if you want to retrain:

```bash
cd model
python -m src.training.train
```

Training takes approximately 30-40 minutes.

---

## 📝 Quick Reference

```bash
# PRIMARY USAGE (Web App)
cd frontend && python app.py
# Open: http://localhost:5001

# Console Testing (Debugging only)
cd model && python -m src.inference.realtime

# Train Model (Optional)
cd model && python -m src.training.train
```

