"""
Transformer-based Face Detection and Emotion Recognition Model
Architecture: CNN Backbone → Transformer Encoder → Decoder → Prediction Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet50
import math


class PositionalEncoding(nn.Module):
    """Add positional encoding to transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1)]


class CNNBackbone(nn.Module):
    """ResNet-50 CNN Backbone for feature extraction"""
    
    def __init__(self, pretrained=True):
        super(CNNBackbone, self).__init__()
        resnet = resnet50(pretrained=pretrained)
        
        # Remove classification head
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Initial convolution
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
    
    def forward(self, x):
        """
        x: (batch, 3, H, W)
        Returns: (batch, 2048, H/32, W/32)
        """
        x = self.conv1(x)  # (batch, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch, 64, H/4, W/4)
        
        x = self.layer1(x)  # (batch, 256, H/4, W/4)
        x = self.layer2(x)  # (batch, 512, H/8, W/8)
        x = self.layer3(x)  # (batch, 1024, H/16, W/16)
        x = self.layer4(x)  # (batch, 2048, H/32, W/32)
        
        return x


class TransformerEncoder(nn.Module):
    """Standard Transformer Encoder"""
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return x


class TransformerDecoder(nn.Module):
    """Standard Transformer Decoder with Object Queries"""
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, num_queries=100):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Learnable object queries
        self.object_queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.pos_encoding = PositionalEncoding(d_model)
    
    def forward(self, memory, mask=None):
        """
        memory: (batch, seq_len, d_model) - encoder output
        Returns: (batch, num_queries, d_model)
        """
        batch_size = memory.size(0)
        queries = self.object_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        queries = self.pos_encoding(queries)
        
        output = self.decoder(queries, memory, memory_key_padding_mask=mask)
        return output


class PredictionHeads(nn.Module):
    """Prediction heads for bounding boxes and class labels"""
    
    def __init__(self, d_model=256, num_classes=1, num_queries=100):
        super(PredictionHeads, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Bounding box regression head (4 coordinates: x, y, w, h)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        # Class prediction head
        self.class_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes + 1)  # +1 for background class
        )
    
    def forward(self, x):
        """
        x: (batch, num_queries, d_model)
        Returns:
            bboxes: (batch, num_queries, 4)
            classes: (batch, num_queries, num_classes+1)
        """
        bboxes = self.bbox_head(x)
        bboxes = torch.sigmoid(bboxes)  # Normalize to [0, 1]
        
        classes = self.class_head(x)
        
        return bboxes, classes


class TransformerDetectionModel(nn.Module):
    """
    Complete Transformer-based Detection Model (DETR-style)
    
    Architecture:
    Image → CNN Backbone → Feature Map
          → Flatten + Projection → Sequence
          → Positional Embedding + Transformer Encoder
          → Transformer Decoder + Object Queries
          → Prediction Heads (class + bbox)
    """
    
    def __init__(self, num_classes=1, num_queries=100, d_model=256, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerDetectionModel, self).__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Step 1: CNN Backbone (ResNet-50)
        self.backbone = CNNBackbone(pretrained=True)
        
        # Step 2: Feature projection layer
        # ResNet-50 outputs 2048 channels, project to d_model
        self.feature_projection = nn.Sequential(
            nn.Conv2d(2048, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model)
        )
        
        # Step 3: Flatten + Positional Embedding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Step 4: Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Step 5: Transformer Decoder + Object Queries
        self.transformer_decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_queries=num_queries
        )
        
        # Step 6: Prediction Heads
        self.prediction_heads = PredictionHeads(
            d_model=d_model,
            num_classes=num_classes,
            num_queries=num_queries
        )
    
    def forward(self, x):
        """
        x: (batch, 3, H, W)
        Returns:
            bboxes: (batch, num_queries, 4) - normalized coordinates [x, y, w, h]
            classes: (batch, num_queries, num_classes+1)
        """
        batch_size = x.size(0)
        
        # Step 1: CNN Backbone - extract features
        features = self.backbone(x)  # (batch, 2048, H/32, W/32)
        
        # Step 2: Project features
        features = self.feature_projection(features)  # (batch, d_model, H/32, W/32)
        
        # Step 3: Flatten to sequence
        b, c, h, w = features.shape
        features_flat = features.flatten(2).transpose(1, 2)  # (batch, h*w, d_model)
        
        # Step 4: Add positional encoding
        features_flat = self.pos_encoding(features_flat)
        
        # Step 5: Transformer Encoder
        encoder_output = self.transformer_encoder(features_flat)  # (batch, h*w, d_model)
        
        # Step 6: Transformer Decoder with object queries
        decoder_output = self.transformer_decoder(encoder_output)  # (batch, num_queries, d_model)
        
        # Step 7: Prediction heads
        bboxes, classes = self.prediction_heads(decoder_output)
        
        return bboxes, classes


class TransformerEmotionModel(nn.Module):
    """
    Transformer-based Emotion Recognition from Face Crops
    """
    
    def __init__(self, num_emotions=8, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super(TransformerEmotionModel, self).__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(d_model * 16, 512),  # 16 = 4*4 spatial positions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, x):
        """
        x: (batch, 3, 224, 224) - face crop
        Returns: (batch, num_emotions) - emotion logits
        """
        # Feature extraction
        features = self.backbone(x)  # (batch, d_model, 4, 4)
        
        # Flatten to sequence
        b, c, h, w = features.shape
        features_flat = features.flatten(2).transpose(1, 2)  # (batch, h*w, d_model)
        
        # Add positional encoding
        features_flat = self.pos_encoding(features_flat)
        
        # Transformer encoder
        transformer_output = self.transformer(features_flat)  # (batch, h*w, d_model)
        
        # Flatten and classify
        transformer_output = transformer_output.flatten(1)  # (batch, h*w*d_model)
        emotion_logits = self.emotion_head(transformer_output)
        
        return emotion_logits


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[*] Testing Transformer Detection Model...")
    detection_model = TransformerDetectionModel(num_classes=1, num_queries=100).to(device)
    x = torch.randn(2, 3, 512, 512).to(device)
    bboxes, classes = detection_model(x)
    print(f"  Bboxes shape: {bboxes.shape}")  # (2, 100, 4)
    print(f"  Classes shape: {classes.shape}")  # (2, 100, 2)
    
    print("[*] Testing Transformer Emotion Model...")
    emotion_model = TransformerEmotionModel(num_emotions=8).to(device)
    x_face = torch.randn(2, 3, 224, 224).to(device)
    emotion_logits = emotion_model(x_face)
    print(f"  Emotion logits shape: {emotion_logits.shape}")  # (2, 8)
    
    print("[+] All models working correctly!")
