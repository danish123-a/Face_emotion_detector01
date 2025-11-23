"""
Transformer-based Face Detection and Emotion Recognition System
Main script using DETR-style architecture
"""

import argparse
import os
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from transformer_model import TransformerDetectionModel, TransformerEmotionModel
from utils.datasets import LoadStreams, LoadImages
from utils.general import create_folder
from utils.torch_utils import select_device


class TransformerFaceDetector:
    """Transformer-based face detection"""
    
    def __init__(self, weights_path="weights/transformer_detection.pt", device='cpu', num_queries=100):
        self.device = torch.device(device)
        self.num_queries = num_queries
        
        # Initialize model
        self.model = TransformerDetectionModel(num_classes=1, num_queries=num_queries).to(self.device)
        self.model.eval()
        
        # Load weights if available
        if os.path.exists(weights_path):
            print(f"[*] Loading detection weights from {weights_path}")
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
    
    def detect(self, image, conf_thres=0.5):
        """
        Detect faces in image
        
        Args:
            image: numpy array (H, W, 3) or filepath
            conf_thres: confidence threshold
        
        Returns:
            list of detections: [x, y, w, h, confidence]
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        # Resize to 512x512
        image_resized = cv2.resize(image, (512, 512))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            bboxes, classes = self.model(image_tensor)  # (1, num_queries, 4), (1, num_queries, 2)
        
        # Extract detections
        class_probs = torch.softmax(classes[0], dim=1)  # (num_queries, 2)
        face_probs = class_probs[:, 0]  # Probability of face class
        
        detections = []
        for i, conf in enumerate(face_probs):
            if conf > conf_thres:
                # Bbox is normalized [0, 1]
                x_norm, y_norm, w_norm, h_norm = bboxes[0, i].cpu().numpy()
                
                # Convert to pixel coordinates
                x1 = int((x_norm - w_norm/2) * 512)
                y1 = int((y_norm - h_norm/2) * 512)
                x2 = int((x_norm + w_norm/2) * 512)
                y2 = int((y_norm + h_norm/2) * 512)
                
                # Scale to original image size
                scale_x = w / 512
                scale_y = h / 512
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, conf.item()])
        
        return detections


class TransformerEmotionClassifier:
    """Transformer-based emotion classification"""
    
    def __init__(self, weights_path="weights/transformer_emotion.pt", device='cpu', num_emotions=8):
        self.device = torch.device(device)
        self.num_emotions = num_emotions
        self.emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        
        # Initialize model
        self.model = TransformerEmotionModel(num_emotions=num_emotions).to(self.device)
        self.model.eval()
        
        # Load weights if available
        if os.path.exists(weights_path):
            print(f"[*] Loading emotion weights from {weights_path}")
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
    
    def classify(self, face_crop):
        """
        Classify emotion from face crop
        
        Args:
            face_crop: numpy array (H, W, 3) or PIL Image
        
        Returns:
            emotion: str (emotion label)
            confidence: float
        """
        # Preprocess
        if isinstance(face_crop, np.ndarray):
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop = Image.fromarray(face_crop)
        
        face_crop = face_crop.resize((224, 224))
        face_tensor = torch.from_numpy(np.array(face_crop)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        face_tensor = face_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            emotion_logits = self.model(face_tensor)  # (1, 8)
        
        # Get prediction
        emotion_probs = torch.softmax(emotion_logits[0], dim=0)
        emotion_idx = emotion_probs.argmax().item()
        confidence = emotion_probs[emotion_idx].item()
        
        emotion_label = self.emotions[emotion_idx]
        
        return emotion_label, confidence


def detect_faces_and_emotions(image_path, detector, emotion_classifier, conf_thres=0.5, show_conf=True):
    """
    Detect faces and emotions in image
    
    Returns:
        annotated_image: numpy array with bounding boxes and emotion labels
        detections: list of detected faces with emotions
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Could not load image from {image_path}")
        return None, []
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Detect faces
    detections = detector.detect(image_rgb, conf_thres=conf_thres)
    
    # Classify emotions and draw
    detections_with_emotions = []
    for x1, y1, x2, y2, conf in detections:
        # Extract face crop
        face_crop = image[y1:y2, x1:x2]
        
        # Classify emotion
        emotion, emotion_conf = emotion_classifier.classify(face_crop)
        
        detections_with_emotions.append({
            'bbox': [x1, y1, x2, y2],
            'face_conf': conf,
            'emotion': emotion,
            'emotion_conf': emotion_conf
        })
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw emotion label
        label = f"{emotion}"
        if show_conf:
            label += f" ({emotion_conf:.2f})"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Draw label background
        cv2.rectangle(image, (x1, y1 - text_size[1] - 10),
                     (x1 + text_size[0] + 10, y1), color, -1)
        
        # Draw text
        cv2.putText(image, label, (x1 + 5, y1 - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    return image, detections_with_emotions


def main(opt):
    """Main detection loop"""
    
    print("[*] Initializing Transformer models...")
    device = select_device(opt.device)
    
    # Initialize detector and emotion classifier
    detector = TransformerFaceDetector(device=device)
    emotion_classifier = TransformerEmotionClassifier(device=device)
    
    print("[+] Models loaded successfully!")
    
    # Source handling
    source = opt.source
    
    if source.isdigit():
        # Webcam
        print(f"[*] Opening webcam {source}...")
        cap = cv2.VideoCapture(int(source))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect(frame_rgb, conf_thres=opt.conf_thres)
            
            # Classify emotions
            for x1, y1, x2, y2, conf in detections:
                face_crop = frame[y1:y2, x1:x2]
                emotion, emotion_conf = emotion_classifier.classify(face_crop)
                
                # Draw
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{emotion} ({emotion_conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show FPS if requested
            if opt.show_fps:
                cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Transformer Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        # Image file
        print(f"[*] Processing image: {source}")
        result_image, detections = detect_faces_and_emotions(
            source, detector, emotion_classifier,
            conf_thres=opt.conf_thres, show_conf=not opt.hide_conf
        )
        
        if result_image is not None:
            # Display
            cv2.imshow("Transformer Emotion Detection", result_image)
            print(f"[+] Detected {len(detections)} faces")
            for i, det in enumerate(detections):
                print(f"  Face {i+1}: {det['emotion']} (conf: {det['emotion_conf']:.2f})")
            
            # Save if requested
            if opt.output_path:
                cv2.imwrite(opt.output_path, result_image)
                print(f"[+] Saved result to {opt.output_path}")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-based Face Detection and Emotion Recognition")
    parser.add_argument('--source', type=str, default='0', help='Input source (0=webcam, image path, or video path)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='', help='Device (cuda or cpu)')
    parser.add_argument('--show-fps', action='store_true', help='Show FPS counter')
    parser.add_argument('--hide-conf', action='store_true', help='Hide confidence scores')
    parser.add_argument('--output-path', type=str, help='Path to save output image')
    
    opt = parser.parse_args()
    main(opt)
