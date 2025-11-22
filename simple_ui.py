#!/usr/bin/env python3
"""
Simple Emotion Detection Gradio UI with Webcam
Welcome to Emotion Detector - Click button to start webcam
"""
import gradio as gr
import cv2
import torch
import numpy as np
from PIL import Image

from emotion import init, detect_emotion
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Global variables
detector = None
device = None
img_size = 512

def initialize_detector():
    """Initialize the face detector model"""
    global detector, device, img_size
    print("[*] Loading face detector...")
    device = select_device('')
    init(device)
    detector = attempt_load("weights/yolov7-tiny.pt", map_location=device)
    stride = int(detector.stride.max().item())
    img_size = check_img_size(512, s=stride)
    detector.eval()
    if device.type != 'cpu':
        detector.half()
    print("[+] Face detector loaded!")

initialize_detector()

def detect_faces_and_emotions(frame):
    """Detect faces and emotions in a frame"""
    global detector, device, img_size
    
    if frame is None:
        return None
    
    try:
        # Ensure frame is in correct format
        if isinstance(frame, list):
            im0 = frame[0] if len(frame) > 0 else None
            if im0 is None:
                return None
        else:
            im0 = frame
        
        if isinstance(im0, np.ndarray):
            im0 = im0.copy()
        else:
            im0 = np.array(im0)
        
        if im0.dtype != np.uint8:
            if im0.max() <= 1.0:
                im0 = (im0 * 255).astype(np.uint8)
            else:
                im0 = im0.astype(np.uint8)
        
        if len(im0.shape) == 2:
            im0 = cv2.cvtColor(im0, cv2.COLOR_GRAY2BGR)
        elif im0.shape[2] == 4:
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGBA2BGR)
        
        h, w = im0.shape[:2]
        
        # Prepare image for inference
        img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if device.type != 'cpu' else img.float()
        img /= 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        # Detect faces
        with torch.no_grad():
            pred = detector(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.5, 0.45, agnostic=False)
        
        # Process detections
        if pred[0] is not None and len(pred[0]) > 0:
            det = pred[0]
            det[:, :4] = scale_coords((img_size, img_size), det[:, :4], im0.shape).round()
            
            face_images = []
            boxes = []
            
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                face_crop = im0[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_images.append(np.array(face_rgb))
                    boxes.append([x1, y1, x2, y2])
            
            if face_images:
                emotion_results = detect_emotion(face_images, conf=True)
                
                for (emotion_label, emotion_idx), (x1, y1, x2, y2) in zip(emotion_results, boxes):
                    color = (0, 255, 0)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                    
                    label = emotion_label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                    
                    cv2.rectangle(im0, (x1, y1 - text_size[1] - 10), 
                                 (x1 + text_size[0] + 10, y1), color, -1)
                    cv2.putText(im0, label, (x1 + 5, y1 - 5), 
                              font, font_scale, (255, 255, 255), thickness)
        
        return im0
    
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return frame

# Create Gradio UI
with gr.Blocks() as demo:
    
    gr.HTML("<h1 style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px;'>EMOTION DETECTOR - Welcome</h1>")
    
    gr.Markdown("### Instructions: Click the button below to enable your webcam and detect emotions!")
    gr.Markdown("Supported Emotions: Anger | Contempt | Disgust | Fear | Happy | Neutral | Sad | Surprise")
    
    with gr.Row():
        video_input = gr.Video(
            sources=["webcam"],
            label="Your Webcam - Click to Enable",
            streaming=True,
            scale=1
        )
        
        video_output = gr.Image(
            label="Emotion Detection Results",
            type="numpy",
            scale=1
        )
    
    # Real-time processing
    video_input.change(
        fn=detect_faces_and_emotions,
        inputs=[video_input],
        outputs=[video_output]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("[*] EMOTION DETECTION - SIMPLE WEBCAM UI")
    print("="*60)
    print("\n[+] Starting Gradio server...\n")
    print("[*] Open your browser at: http://localhost:7860")
    print("\n" + "="*60 + "\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
