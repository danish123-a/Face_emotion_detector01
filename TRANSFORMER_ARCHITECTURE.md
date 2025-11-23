# Transformer-Based Face Detection and Emotion Recognition

## Architecture Overview

### Complete Pipeline:
```
Image (RGB)
    ↓
[CNN Backbone - ResNet-50]
    ↓
Feature Maps (2048 channels, H/32 × W/32)
    ↓
[Feature Projection Layer]
    ↓
Flattened Sequence + Positional Embedding
    ↓
[Transformer Encoder (6 layers)]
- Multi-head Self-Attention
- Feed-forward Networks
- Layer Normalization
    ↓
Encoded Features
    ↓
[Transformer Decoder (6 layers)]
- Multi-head Cross-Attention
- Object Queries (learnable, 100 queries)
- Feed-forward Networks
    ↓
Decoded Representations
    ↓
[Prediction Heads]
├─ Bounding Box Regression Head (4 values: x, y, w, h)
└─ Class Classification Head (2 classes: face/background)
    ↓
Final Detections (bboxes + confidence scores)
```

## Key Components

### 1. **CNN Backbone (ResNet-50)**
- **Input**: RGB image (3, 512, 512)
- **Output**: Feature maps (2048, 16, 16)
- **Purpose**: Extract semantic features from images
- **Advantages**:
  - Pre-trained on ImageNet
  - Robust feature extraction
  - Well-established architecture

### 2. **Feature Projection Layer**
- **Input**: ResNet-50 features (2048 channels)
- **Output**: Projected features (256 channels)
- **Purpose**: Reduce dimensionality for transformer efficiency

### 3. **Positional Encoding**
- **Type**: Sinusoidal positional encoding
- **Purpose**: Add spatial information to sequence
- **Formula**:
  ```
  PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```

### 4. **Transformer Encoder**
- **Input**: Flattened features + positional encoding
- **Layers**: 6 layers of transformer encoder blocks
- **Head**: 8 attention heads
- **Dimension**: 256-dimensional embeddings
- **Purpose**: Learn global context and relationships

### 5. **Transformer Decoder**
- **Input**: Encoder output + learned object queries
- **Object Queries**: 100 learnable query vectors
- **Layers**: 6 layers of transformer decoder blocks
- **Purpose**: Decode encoder features into predictions

### 6. **Prediction Heads**
#### Bounding Box Head:
- Input: (batch, 100, 256)
- Output: (batch, 100, 4) - normalized coordinates [x, y, w, h]
- Activation: Sigmoid (outputs normalized to [0, 1])

#### Class Head:
- Input: (batch, 100, 256)
- Output: (batch, 100, 2) - logits for face/background classes
- Activation: None (logits, used with CrossEntropyLoss)

## Emotion Recognition Architecture

### TransformerEmotionModel:
```
Face Crop (3, 224, 224)
    ↓
[CNN Feature Extractor]
- Conv2d layers with BatchNorm
- Spatial reduction to 4×4
    ↓
Flattened Features (16, 256)
    ↓
[Transformer Encoder (4 layers)]
- Multi-head self-attention
- Context aggregation
    ↓
[Classification Head]
- Linear layers
- Dropout
    ↓
Emotion Logits (8 emotions)
    ↓
Output: emotion label + confidence
```

## Advantages over YOLOv7

| Aspect | YOLOv7 | Transformer |
|--------|--------|-------------|
| **Architecture** | CNN-based, grid-based | Attention-based, query-based |
| **Scalability** | Limited to grid cells | Flexible object queries |
| **Context** | Local receptive fields | Global attention |
| **Speed** | Faster inference | Slower but more accurate |
| **Flexibility** | Fixed output format | Adaptable output format |
| **Learning** | Direct coordinate regression | Learns relationships via attention |

## Files Created

### Core Models:
- **`transformer_model.py`**: Contains all transformer architecture components
  - `TransformerDetectionModel`: DETR-style face detector
  - `TransformerEmotionModel`: Transformer-based emotion classifier
  - Supporting modules: `CNNBackbone`, `TransformerEncoder`, `TransformerDecoder`, `PositionalEncoding`, etc.

### Main Scripts:
- **`main_transformer.py`**: Complete inference pipeline
  - `TransformerFaceDetector`: Detection wrapper class
  - `TransformerEmotionClassifier`: Emotion classification wrapper class
  - `detect_faces_and_emotions()`: Full detection + emotion pipeline
  - CLI interface for webcam, image, or video input

### Training:
- **`train_transformer.py`**: Training script for both models
  - Dummy datasets for demonstration
  - Training loops with loss computation
  - Model checkpointing
  - Accuracy metrics for emotion classification

## Usage

### Install Dependencies:
```bash
pip install torch torchvision
```

### 1. Inference with Transformer Models:

**Webcam Detection:**
```bash
python main_transformer.py --source 0 --show-fps
```

**Image Processing:**
```bash
python main_transformer.py --source image.jpg --output-path result.jpg
```

**With Confidence Scores:**
```bash
python main_transformer.py --source 0 --conf-thres 0.5
```

### 2. Training Custom Models:

**Train Detection Model:**
```bash
python train_transformer.py --model detection --epochs 20 --batch-size 8
```

**Train Emotion Model:**
```bash
python train_transformer.py --model emotion --epochs 20 --batch-size 32
```

**Train Both:**
```bash
python train_transformer.py --model both --epochs 20 --batch-size 8
```

### 3. Using Models Programmatically:

```python
from transformer_model import TransformerDetectionModel, TransformerEmotionModel
import torch

# Initialize models
detector = TransformerDetectionModel(num_classes=1, num_queries=100)
emotion_model = TransformerEmotionModel(num_emotions=8)

# Load pre-trained weights
detector.load_state_dict(torch.load('weights/transformer_detection.pt'))
emotion_model.load_state_dict(torch.load('weights/transformer_emotion.pt'))

# Inference
image = torch.randn(1, 3, 512, 512)
bboxes, classes = detector(image)  # (1, 100, 4), (1, 100, 2)

face_crop = torch.randn(1, 3, 224, 224)
emotions = emotion_model(face_crop)  # (1, 8)
```

## Model Specifications

### Transformer Detection Model:
- **Input**: (batch, 3, 512, 512)
- **Output**: 
  - Bounding boxes: (batch, 100, 4)
  - Classes: (batch, 100, 2)
- **Parameters**: ~55M
- **Memory**: ~3GB (batch_size=2)

### Transformer Emotion Model:
- **Input**: (batch, 3, 224, 224)
- **Output**: (batch, 8) - emotion logits
- **Parameters**: ~10M
- **Memory**: ~1GB (batch_size=32)

## Training Considerations

### Dataset Format:
The training script expects:
- **Detection**: Images with bounding box annotations and class labels
- **Emotion**: Face crops with emotion labels (0-7)

### Replace Dummy Datasets:
Edit `train_transformer.py`:
```python
# Replace DummyDetectionDataset with your own
class YourDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        # Load images and annotations
        pass
    
    def __getitem__(self, idx):
        # Return image, bboxes, classes
        return image, bboxes, classes

# Replace DummyEmotionDataset with your own
class YourEmotionDataset(Dataset):
    def __init__(self, face_dir, label_file):
        # Load face crops and labels
        pass
    
    def __getitem__(self, idx):
        # Return face, emotion_label
        return face, emotion
```

## Hyperparameters

### Detection Model:
- `num_queries`: 100 (number of object queries)
- `d_model`: 256 (embedding dimension)
- `nhead`: 8 (number of attention heads)
- `num_encoder_layers`: 6
- `num_decoder_layers`: 6
- `dim_feedforward`: 2048
- `dropout`: 0.1

### Emotion Model:
- `d_model`: 256
- `nhead`: 8
- `num_layers`: 4
- `dim_feedforward`: 1024

## Performance Notes

- **Speed**: Transformer models are slower than YOLOv7 (~50-100ms per image on CPU)
- **Accuracy**: Generally more accurate, especially for multiple faces
- **Memory**: Requires more GPU memory than YOLOv7
- **Training**: Requires larger datasets for good performance

## Next Steps

1. **Prepare Real Dataset**:
   - Collect face detection dataset (WIDER Face, AFW, etc.)
   - Collect emotion dataset (FER2013, AffectNet, etc.)

2. **Fine-tune Models**:
   - Modify dataset loaders in `train_transformer.py`
   - Adjust hyperparameters
   - Train on your data

3. **Optimize for Deployment**:
   - Convert to ONNX format
   - Quantize models
   - Deploy to edge devices

4. **Evaluate Performance**:
   - Benchmark on test sets
   - Compare with YOLOv7 baseline
   - Calculate metrics (mAP, F1-score, etc.)

## References

- DETR: End-to-End Object Detection with Transformers (https://arxiv.org/abs/2005.12138)
- Attention Is All You Need (https://arxiv.org/abs/1706.03762)
- ResNet: Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

## License

This transformer architecture is provided as-is for educational and research purposes.
