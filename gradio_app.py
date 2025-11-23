import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import torch
import warnings
from main import detect_emotion_from_image
from emotion import init
from utils.torch_utils import select_device
import tempfile
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize device and model once
device = select_device('')
init(device)

# CSS styling for the app
custom_css = """
.title-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.title-text {
    color: white;
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    margin: 0;
    letter-spacing: 1px;
}

.subtitle-text {
    color: rgba(255, 255, 255, 0.9);
    font-size: 14px;
    text-align: center;
    margin-top: 10px;
}

.tab-content {
    padding: 20px;
    border-radius: 10px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.result-box {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.emotion-text {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    margin: 10px 0;
}

.emotion-happy {
    color: #FFD700;
}

.emotion-sad {
    color: #4169E1;
}

.emotion-angry {
    color: #DC143C;
}

.emotion-surprise {
    color: #FF8C00;
}

.emotion-fear {
    color: #8B008B;
}

.emotion-disgust {
    color: #228B22;
}

.emotion-neutral {
    color: #808080;
}
"""

# Process webcam input - Real-time frame processing
def process_webcam(frame):
    """Process a single frame from webcam with emotion detection"""
    if frame is None:
        return None, "<div class='result-box'><p style='color: orange;'>No frame captured. Please allow camera access.</p></div>"
    
    try:
        # Ensure frame is numpy array
        if isinstance(frame, str):
            frame = cv2.imread(frame)
        
        if frame is None or frame.size == 0:
            return None, "<div class='result-box'><p style='color: red;'>Error: Invalid frame data</p></div>"
        
        # Save frame temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            tmp_path = tmp.name
        
        print(f"[WEBCAM] Processing frame: {tmp_path}")
        
        try:
            result = detect_emotion_from_image(tmp_path, device=device)
            print(f"[WEBCAM] Result: {result['success']}")
            
            if result['success']:
                output_image = result['image']
                emotions = result['emotions']
                
                # Convert BGR to RGB for Gradio display
                if output_image is not None:
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                
                if emotions:
                    emotion_text = "<div class='result-box'><h3>Detected Emotions:</h3>"
                    for i, emotion in enumerate(emotions, 1):
                        emotion_text += f"<p class='emotion-text emotion-{emotion['label'].lower()}'>Face {i}: <b>{emotion['label'].upper()}</b></p>"
                    emotion_text += "</div>"
                else:
                    emotion_text = "<div class='result-box'><p style='color: orange;'>No faces detected in this frame</p></div>"
                
                return output_image, emotion_text
            else:
                # Convert frame to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb, f"<div class='result-box'><p style='color: red;'>Error: {result['error']}</p></div>"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"[WEBCAM] Exception: {str(e)}")
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
        except:
            frame_rgb = None
        return frame_rgb, f"<div class='result-box'><p style='color: red;'>Exception: {str(e)}</p></div>"


# Process uploaded image
def process_image(image):
    """Process uploaded image"""
    if image is None:
        return None, "<div class='result-box'><p style='color: orange;'>No image uploaded</p></div>"
    
    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name
    
    try:
        result = detect_emotion_from_image(tmp_path, device=device)
        if result['success']:
            output_image = result['image']
            emotions = result['emotions']
            
            # Convert BGR to RGB for Gradio display
            if output_image is not None:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            
            emotion_text = "<div class='result-box'><h3>Detected Emotions:</h3>"
            if emotions:
                for i, emotion in enumerate(emotions, 1):
                    emotion_text += f"<p class='emotion-text emotion-{emotion['label'].lower()}'>Face {i}: <b>{emotion['label'].upper()}</b></p>"
            else:
                emotion_text += "<p style='color: orange;'>No faces detected in the image</p>"
            emotion_text += "</div>"
            
            return output_image, emotion_text
        else:
            # Convert image to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
            return image_rgb, f"<div class='result-box'><p style='color: red;'>Error: {result['error']}</p></div>"
    except Exception as e:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None
        return image_rgb, f"<div class='result-box'><p style='color: red;'>Exception: {str(e)}</p></div>"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Process uploaded video - Frame by frame analysis
def process_video(video_path):
    """Process uploaded video and analyze emotions in each frame"""
    if video_path is None:
        return None, "<div class='result-box'><p style='color: orange;'>No video uploaded</p></div>"
    
    try:
        print(f"[VIDEO] Input type: {type(video_path)}, Value: {video_path}")
        
        # Handle both string paths and dict with 'name' key
        if isinstance(video_path, dict):
            video_file = video_path.get('name') or video_path.get('path')
        else:
            video_file = str(video_path)
        
        print(f"[VIDEO] Using file: {video_file}")
        
        # Open video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None, "<div class='result-box'><p style='color: red;'>Error: Cannot open video file</p></div>"
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[VIDEO] Video properties - FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
        
        # Create temporary video file for output
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp_video:
            output_video_path = tmp_video.name
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        emotion_summary = {}
        frames_with_detections = 0
        
        print(f"[VIDEO] Starting video processing: {total_frames} total frames")
        
        # Process every frame
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame for emotion detection
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                tmp_path = tmp.name
            
            processed_frame = frame.copy()
            
            try:
                result = detect_emotion_from_image(tmp_path, device=device)
                if result['success']:
                    annotated_frame = result['image']
                    emotions = result['emotions']
                    
                    # Track emotions detected
                    if emotions:
                        frames_with_detections += 1
                        for emotion in emotions:
                            label = emotion['label']
                            emotion_summary[label] = emotion_summary.get(label, 0) + 1
                    
                    # Write annotated frame to output video
                    if annotated_frame is not None:
                        processed_frame = annotated_frame
                    
            except Exception as e:
                print(f"[VIDEO] Error processing frame {frame_count}: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            # Write frame to output video
            out.write(processed_frame)
            frame_count += 1
            
            # Progress indicator
            if frame_count % 10 == 0:
                print(f"[VIDEO] Processing video: {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"[VIDEO] Processing complete!")
        
        # Generate results summary
        emotion_text = "<div class='result-box'>"
        emotion_text += "<h3>Video Analysis Complete</h3>"
        emotion_text += f"<p><b>Total Frames Processed:</b> {frame_count}</p>"
        emotion_text += f"<p><b>Frames with Face Detected:</b> {frames_with_detections}</p>"
        
        if emotion_summary:
            emotion_text += "<p><b>Emotion Summary:</b></p>"
            for emotion, count in sorted(emotion_summary.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / frames_with_detections * 100) if frames_with_detections > 0 else 0
                emotion_text += f"<p class='emotion-text emotion-{emotion.lower()}'>{emotion.upper()}: {count} frames ({percentage:.1f}%)</p>"
        else:
            emotion_text += "<p style='color: orange;'>No faces detected in any frames</p>"
        
        emotion_text += "</div>"
        
        print(f"[VIDEO] Output saved to: {output_video_path}")
        return output_video_path, emotion_text
    
    except Exception as e:
        print(f"[VIDEO] Major exception: {str(e)}")
        return None, f"<div class='result-box'><p style='color: red;'>Error: {str(e)}</p></div>"


# Create Gradio Interface with nested templates
def create_interface():
    with gr.Blocks(css=custom_css) as demo:
        
        # Main Title with styling
        with gr.Group():
            gr.HTML("""
                <div class="title-container">
                    <h1 class="title-text">Emotion Detector with Face</h1>
                    <p class="subtitle-text">Detect emotions from your face using advanced AI technology</p>
                </div>
            """)
        
        # Create tabs inside template
        with gr.Tabs():
            
            # ============ TAB 1: WEBCAM & IMAGE & VIDEO ============
            with gr.Tab("Input Methods"):
                gr.HTML("<div class='tab-content'><h2 style='text-align: center; color: #667eea;'>Choose Your Input Method</h2></div>")
                
                with gr.Tabs():
                    
                    # Webcam Tab
                    with gr.Tab("Webcam"):
                        gr.HTML("<div style='padding: 15px; background: #e3f2fd; border-radius: 8px; margin-bottom: 20px;'><b>How to use:</b> Allow camera access and click 'Analyze' to detect emotions in real-time</div>")
                        
                        with gr.Row():
                            with gr.Column():
                                webcam_input = gr.Image(sources=["webcam"], type="numpy", label="Webcam Input")
                                webcam_btn = gr.Button("Analyze Webcam", variant="primary", size="lg")
                            
                            with gr.Column():
                                webcam_output_image = gr.Image(label="Processed Image", type="numpy")
                                webcam_output_text = gr.HTML(label="Results")
                        
                        webcam_btn.click(
                            process_webcam,
                            inputs=[webcam_input],
                            outputs=[webcam_output_image, webcam_output_text]
                        )
                    
                    # Image Upload Tab
                    with gr.Tab("Upload Image"):
                        gr.HTML("<div style='padding: 15px; background: #f3e5f5; border-radius: 8px; margin-bottom: 20px;'><b>How to use:</b> Upload an image with faces to detect emotions</div>")
                        
                        with gr.Row():
                            with gr.Column():
                                image_input = gr.Image(type="numpy", label="Upload Image")
                                
                                with gr.Row():
                                    image_btn = gr.Button("Upload Image", variant="primary", size="lg", scale=2)
                                    image_clear_btn = gr.Button("Delete", variant="stop", size="lg", scale=1)
                            
                            with gr.Column():
                                image_output_image = gr.Image(label="Processed Image", type="numpy")
                                image_output_text = gr.HTML(label="Results")
                        
                        image_btn.click(
                            process_image,
                            inputs=[image_input],
                            outputs=[image_output_image, image_output_text]
                        )
                        
                        # Clear/Delete function
                        def clear_image():
                            return None, None, "<div class='result-box'><p style='color: green;'>Image cleared</p></div>"
                        
                        image_clear_btn.click(
                            clear_image,
                            inputs=[],
                            outputs=[image_input, image_output_image, image_output_text]
                        )
                    
                    # Video Upload Tab
                    with gr.Tab("Upload Video"):
                        gr.HTML("<div style='padding: 15px; background: #e8f5e9; border-radius: 8px; margin-bottom: 20px;'><b>How to use:</b> Upload a video to detect emotions in each frame. Output video shows annotations with emotions.</div>")
                        
                        with gr.Row():
                            with gr.Column():
                                video_input = gr.Video(label="Upload Video")
                                video_btn = gr.Button("Analyze Video", variant="primary", size="lg")
                            
                            with gr.Column():
                                video_output_video = gr.Video(label="Processed Video with Annotations", interactive=False)
                                video_output_text = gr.HTML(label="Results")
                        
                        video_btn.click(
                            process_video,
                            inputs=[video_input],
                            outputs=[video_output_video, video_output_text]
                        )
            
            # ============ TAB 2: ABOUT & INFO ============
            with gr.Tab("About"):
                with gr.Group():
                    gr.HTML("""
                        <div class="tab-content">
                            <div class="result-box">
                                <h2 style="color: #667eea;">About This Application</h2>
                                <p><b>Emotion Detector with Face</b> uses advanced AI to detect emotions in real-time.</p>
                                
                                <h3 style="color: #764ba2;">Features:</h3>
                                <ul style="font-size: 16px; line-height: 1.8;">
                                    <li>Real-time webcam emotion detection</li>
                                    <li>Image-based emotion analysis</li>
                                    <li>Video frame-by-frame emotion tracking with annotations</li>
                                    <li>Multiple emotion recognition</li>
                                    <li>High-accuracy face detection with YOLOv7</li>
                                </ul>
                                
                                <h3 style="color: #764ba2;">Supported Emotions:</h3>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                    <p><span style="color: #FFD700;">Happy</span></p>
                                    <p><span style="color: #4169E1;">Sad</span></p>
                                    <p><span style="color: #DC143C;">Angry</span></p>
                                    <p><span style="color: #FF8C00;">Surprise</span></p>
                                    <p><span style="color: #8B008B;">Fear</span></p>
                                    <p><span style="color: #228B22;">Disgust</span></p>
                                    <p><span style="color: #808080;">Neutral</span></p>
                                </div>
                                
                                <h3 style="color: #764ba2;">Tips for Best Results:</h3>
                                <ul style="font-size: 16px; line-height: 1.8;">
                                    <li>Ensure good lighting</li>
                                    <li>Face should be clearly visible</li>
                                    <li>Use high-resolution images or videos</li>
                                    <li>Keep distance of 30cm - 1m from camera</li>
                                </ul>
                            </div>
                        </div>
                    """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        show_api=False
    )
