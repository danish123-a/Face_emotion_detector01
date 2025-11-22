#!/usr/bin/env python3
"""
🎭 EMOTION DETECTION - QUICK START GUIDE
========================================

This script shows you how to run the Emotion Detection system
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║           🎭 EMOTION DETECTION - QUICK START GUIDE                ║
╚════════════════════════════════════════════════════════════════════╝

✅ PROJECT SETUP COMPLETE!

You now have 3 ways to use the emotion detection system:

1️⃣  GRADIO WEB UI (EASIEST & BEST FOR SHARING)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Option A - Windows Users:
     Double-click: start_ui.bat
   
   Option B - Command Line:
     python app.py
   
   Then open browser to: http://localhost:7860
   
   Features:
   ✓ Upload images for emotion detection
   ✓ Real-time webcam detection
   ✓ Beautiful web interface
   ✓ Shareable link with friends
   ✓ Works on any device with a browser


2️⃣  WEBCAM DETECTION (FOR REAL-TIME USE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Option A - Windows Users:
     Double-click: run_webcam.bat
   
   Option B - Command Line:
     python main.py --source 0 --show-fps
   
   Features:
   ✓ Opens your webcam
   ✓ Real-time face detection
   ✓ Emotion classification
   ✓ FPS counter
   ✓ Press 'q' to quit


3️⃣  IMAGE PROCESSING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   python main.py --source image.jpg --output-path result.jpg
   
   Features:
   ✓ Process any image file
   ✓ Save annotated results
   ✓ Detects multiple faces


📚 ALL AVAILABLE OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   --source SOURCE              Input (0=webcam, image.jpg, video.mp4)
   --img-size IMG_SIZE          Inference size (default: 512)
   --conf-thres CONF_THRES      Face confidence threshold (default: 0.5)
   --iou-thres IOU_THRES        IOU threshold for NMS (default: 0.45)
   --device DEVICE              Device to use (cpu or 0,1,2...)
   --output-path PATH           Save location
   --show-fps                   Show FPS in console
   --hide-conf                  Hide confidence scores


🎭 SUPPORTED EMOTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   😠 Anger       😒 Contempt    🤢 Disgust     😨 Fear
   😊 Happy       😐 Neutral     😢 Sad         😮 Surprise


📊 PROJECT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   emotion/
   ├── app.py                  ← Gradio UI
   ├── main.py                 ← Main detection script
   ├── emotion.py              ← Emotion model
   ├── start_ui.bat            ← Windows launcher
   ├── start_ui.py             ← Python launcher
   ├── run_webcam.bat          ← Webcam launcher
   ├── models/                 ← Detection models
   ├── utils/                  ← Utilities
   └── weights/                ← Pre-trained weights


🚀 QUICK START COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   # Start Gradio UI
   python app.py

   # Test with webcam
   python main.py --source 0

   # Process image
   python main.py --source photo.jpg --output-path result.jpg

   # With GPU
   python main.py --source 0 --device 0

   # With FPS counter
   python main.py --source 0 --show-fps


💡 TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   • Gradio UI (app.py) is the BEST way to share results
   • Use webcam mode for real-time testing
   • Save results with --output-path
   • Use GPU for faster processing
   • Works on both CPU and GPU


❓ TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   If Gradio UI doesn't work:
   1. Open Command Prompt
   2. cd C:\\Users\\danis\\OneDrive\\Desktop\\face_detection_2\\emotion
   3. python app.py
   4. Open http://localhost:7860 in browser

   If webcam doesn't work:
   • Check if camera is connected
   • Try: python main.py --source 0 --device cpu
   • Press 'q' to exit


📞 ENJOY!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Your emotion detection system is ready to use! 🎉
   Start with the Gradio UI for the best experience.

╚════════════════════════════════════════════════════════════════════╝
""")

input("\nPress Enter to close this window...")
