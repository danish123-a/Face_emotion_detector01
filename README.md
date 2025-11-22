# Emotion Detection System - Clean & Minimal Setup

## Overview
A lightweight face detection and emotion recognition system using:
- **YOLOv7-tiny** for face detection (37 MB)
- **RepVGG-A0** for emotion classification (26 MB)
- **PyTorch 2.7.1** as the deep learning framework
- **Gradio** for web UI interface

## Project Structure

```
emotion/
├── main.py              ← Core detection engine
├── emotion.py           ← Emotion model (RepVGG)
├── repvgg.py            ← RepVGG architecture
├── simple_ui.py         ← Gradio web interface
├── run_webcam.py        ← Webcam detection script
├── QUICK_START.py       ← Quick start guide
├── requirements.txt     ← Dependencies
├── models/              ← YOLOv7 model files
│   ├── common.py
│   ├── experimental.py
│   ├── yolo.py
│   └── __init__.py
├── utils/               ← Utility functions
│   ├── datasets.py
│   ├── general.py
│   ├── plots.py
│   ├── torch_utils.py
│   └── __init__.py
└── weights/             ← Pre-trained weights
    ├── yolov7-tiny.pt   (37 MB - Face detector)
    └── repvgg.pth       (26 MB - Emotion classifier)
```

## Quick Start

### 1. Run Gradio Web UI (Best for Sharing)
```bash
python simple_ui.py
```
Then open: **http://localhost:7860**

### 2. Webcam Real-Time Detection
```bash
python main.py --source 0 --show-fps
```

### 3. Process Single Image
```bash
python main.py --source image.jpg --output-path result.jpg
```

## Supported Emotions (8 Classes)
😠 Anger | 😒 Contempt | 🤢 Disgust | 😨 Fear | 😊 Happy | 😐 Neutral | 😢 Sad | 😮 Surprise

## Requirements
- Python 3.8+
- PyTorch 2.0+
- OpenCV
- Gradio
- NumPy, Pandas

See `requirements.txt` for full dependencies.

## Features
✅ Real-time face detection and emotion classification  
✅ Bounding box visualization with emotion labels  
✅ Web UI for easy sharing  
✅ Webcam streaming support  
✅ Image file processing  
✅ Video stream support  
✅ FPS counter  
✅ CPU and GPU support  

## Command-Line Options
```
--source SOURCE          Input (0=webcam, image.jpg, video.mp4)
--img-size SIZE          Inference size (default: 512)
--conf-thres THRESHOLD   Face confidence threshold (default: 0.5)
--iou-thres THRESHOLD    IOU threshold for NMS (default: 0.45)
--device DEVICE          Device (cpu or 0,1,2... for GPU)
--output-path PATH       Save location
--show-fps              Show FPS in console
--hide-conf             Hide confidence scores
```

## Files Removed During Cleanup
- Unnecessary image files (.webp, .mp4)
- Cache folders (__pycache__, .gradio)
- Unused utility modules (google_utils.py, metrics.py, autoanchor.py)
- Old UI files (app.py, start_ui.bat, etc.)
- Documentation (HOW_TO_RUN_UI.txt)

**Result**: Project reduced to ~65 MB (essential files only)

## Running the System

### Option 1: Web UI (Recommended)
```bash
cd emotion
python simple_ui.py
```
Access via browser: http://localhost:7860

### Option 2: Command Line
```bash
# Webcam detection
python main.py --source 0

# Image processing
python main.py --source photo.jpg --output-path result.jpg

# Video processing
python main.py --source video.mp4 --output-path output.mp4
```

### Option 3: Quick Start Guide
```bash
python QUICK_START.py
```

## Troubleshooting

**Webcam not working?**
```bash
python main.py --source 0 --device cpu
```

**GPU not available?**
```bash
python main.py --source 0 --device cpu
```

**Port 7860 already in use?**
Edit `simple_ui.py` and change `server_port=7860` to another port (e.g., 7861)

## Notes
- Models are loaded on startup for faster inference
- First run may take longer as models are loaded into memory
- GPU recommended for real-time webcam detection
- CPU mode works but will be slower

## License
Original YOLOv7: https://github.com/WongKinYiu/yolov7  
RepVGG: https://github.com/DingXiaoH/RepVGG  
Gradio: https://github.com/gradio-app/gradio

---
**Cleaned and optimized for production use** ✨
