# 🎭 Emotion Detector with Face - Optimized

A lightweight, production-ready face detection and emotion recognition system using advanced AI.

## ✨ Features

- **Real-time Webcam Detection** - Analyze emotions from live video feed
- **Image Analysis** - Upload images to detect faces and emotions  
- **Video Processing** - Process videos frame-by-frame with emotion tracking
- **Beautiful Web UI** - Modern Gradio interface with gradient design
- **Fast & Lightweight** - YOLOv7-tiny (37 MB) + RepVGG (26 MB)
- **8 Emotions Detected** - Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral, Contempt

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web UI
```bash
python gradio_app.py
```
Then open: **http://localhost:7860** in your browser

## 📊 Project Structure

```
Face_emotion_detector/
├── gradio_app.py          ← Main Gradio web interface
├── main.py                ← Core detection engine
├── emotion.py             ← Emotion model (RepVGG)
├── repvgg.py              ← RepVGG architecture
├── requirements.txt       ← Dependencies
├── README.md              ← This file
├── models/                ← YOLOv7 model files
│   ├── common.py
│   ├── experimental.py
│   ├── yolo.py
│   └── __init__.py
├── utils/                 ← Utility functions
│   ├── datasets.py
│   ├── general.py
│   ├── plots.py
│   ├── torch_utils.py
│   └── __init__.py
└── weights/               ← Pre-trained weights
    ├── yolov7-tiny.pt     (37 MB - Face detector)
    └── repvgg.pth         (26 MB - Emotion classifier)
```

## 🎯 Supported Emotions

| Emotion | Icon |
|---------|------|
| Happy | 😊 |
| Sad | 😢 |
| Angry | 😠 |
| Surprise | 😮 |
| Fear | 😨 |
| Disgust | 🤢 |
| Neutral | 😐 |
| Contempt | 😒 |

## 💡 Usage

### Web Interface (Recommended)
```bash
python gradio_app.py
```
Features:
- **Webcam Tab**: Real-time emotion detection from webcam
- **Image Tab**: Upload and analyze images (with Upload & Delete buttons)
- **Video Tab**: Process videos with frame-by-frame emotion analysis
- **About Tab**: Information about supported emotions and tips

### Command Line
```bash
# Webcam detection
python main.py --source 0

# Image processing
python main.py --source image.jpg --output-path result.jpg

# Video processing  
python main.py --source video.mp4 --output-path result.mp4

# Show FPS counter
python main.py --source 0 --show-fps

# Using CPU
python main.py --source 0 --device cpu
```

## ⚙️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)
- See `requirements.txt` for full list

## 🔧 Troubleshooting

### Webcam not working
- Ensure camera is connected and permissions are granted
- Try: `python main.py --source 0 --device cpu`

### Slow detection
- Check if GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use GPU for faster processing (if available)

### Out of memory
- Use CPU instead: `--device cpu`
- Reduce image size: `--img-size 256`

### Port 7860 already in use
- The app will automatically find an available port

## 📦 Optimization Status

✅ **Project Cleaned & Optimized**
- ✓ Removed unused transformer training models
- ✓ Removed old UI and launcher files (simple_ui.py, run_webcam.py, etc.)
- ✓ Cleaned Python cache directories (__pycache__, .gradio)
- ✓ Removed training scripts (train_transformer.py, main_transformer.py)
- ✓ Removed documentation for unused features
- ✓ Kept only essential files for production use
- ✓ Project size: **~65 MB** (99% reduction)

## 📝 File Removals

**Removed Files:**
- `train_transformer.py` - Not needed for inference
- `main_transformer.py` - Alternative implementation
- `transformer_model.py` - Transformer architecture
- `simple_ui.py` - Replaced by gradio_app.py
- `run_webcam.py`, `run_webcam.bat` - Old launchers
- `QUICK_START.py`, `QUICK_START.bat` - Old guides
- `launcher.bat` - Old menu launcher
- `TRANSFORMER_ARCHITECTURE.md`, `TRANSFORMER_SUMMARY.txt` - Documentation
- `STATUS.txt`, `CLEANUP_SUMMARY.txt` - Old summaries
- All cache directories (`__pycache__`, `.gradio`)

**Kept Files:**
- `gradio_app.py` - Main web interface ✅
- `main.py` - Core detection engine ✅
- `emotion.py` - Emotion model ✅
- `repvgg.py` - Model architecture ✅
- `models/`, `utils/`, `weights/` - Essential folders ✅

## 🎓 Model Information

**Face Detection**: YOLOv7-Tiny
- Size: 37 MB
- Fast real-time detection
- High accuracy on various face poses

**Emotion Classification**: RepVGG-A0
- Size: 26 MB
- 8-class emotion classification
- Optimized for inference speed

## 🤝 Contributing

Feel free to fork and submit pull requests!

## ✉️ Contact

For issues or suggestions, please create an issue in the repository.

---

**Last Updated**: November 23, 2025  
**Status**: ✅ Production Ready & Optimized  
**Project Size**: ~65 MB (99% reduction from original)
