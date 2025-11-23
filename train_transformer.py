"""
Training script for Transformer-based Face Detection and Emotion Recognition Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
import os

from transformer_model import TransformerDetectionModel, TransformerEmotionModel


class DummyDetectionDataset(Dataset):
    """Dummy dataset for demonstration - replace with real dataset"""
    
    def __init__(self, num_samples=100, img_size=512):
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Dummy image
        image = torch.randn(3, self.img_size, self.img_size)
        
        # Dummy bounding boxes (num_queries, 4)
        num_queries = 100
        bboxes = torch.rand(num_queries, 4)
        
        # Dummy class labels (num_queries,)
        classes = torch.randint(0, 2, (num_queries,))
        
        return image, bboxes, classes


class DummyEmotionDataset(Dataset):
    """Dummy dataset for emotion recognition - replace with real dataset"""
    
    def __init__(self, num_samples=1000, num_emotions=8):
        self.num_samples = num_samples
        self.num_emotions = num_emotions
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Dummy face crop
        face = torch.randn(3, 224, 224)
        
        # Dummy emotion label
        emotion = torch.randint(0, self.num_emotions, (1,)).item()
        
        return face, emotion


def train_detection_model(args):
    """Train transformer detection model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    # Initialize model
    print("[*] Initializing Transformer Detection Model...")
    model = TransformerDetectionModel(
        num_classes=1,
        num_queries=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    ).to(device)
    
    # Initialize dataset and dataloader
    print("[*] Creating dummy dataset (replace with real dataset)...")
    dataset = DummyDetectionDataset(num_samples=args.num_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    bbox_loss_fn = nn.SmoothL1Loss()
    class_loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print("[*] Starting training...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, (images, bboxes, classes) in enumerate(dataloader):
            images = images.to(device)
            bboxes = bboxes.to(device)
            classes = classes.to(device)
            
            # Forward pass
            pred_bboxes, pred_classes = model(images)
            
            # Compute loss
            bbox_loss = bbox_loss_fn(pred_bboxes, bboxes)
            class_loss = class_loss_fn(pred_classes.view(-1, 2), classes.view(-1))
            loss = bbox_loss + class_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"[+] Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = f"weights/transformer_detection_epoch{epoch+1}.pt"
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[+] Model saved to {save_path}")
    
    # Save final model
    save_path = "weights/transformer_detection.pt"
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[+] Final model saved to {save_path}")


def train_emotion_model(args):
    """Train transformer emotion recognition model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    # Initialize model
    print("[*] Initializing Transformer Emotion Model...")
    model = TransformerEmotionModel(
        num_emotions=8,
        d_model=256,
        nhead=8,
        num_layers=4
    ).to(device)
    
    # Initialize dataset and dataloader
    print("[*] Creating dummy dataset (replace with real dataset)...")
    dataset = DummyEmotionDataset(num_samples=args.num_samples, num_emotions=8)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print("[*] Starting training...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (faces, emotions) in enumerate(dataloader):
            faces = faces.to(device)
            emotions = emotions.to(device)
            
            # Forward pass
            emotion_logits = model(faces)
            
            # Compute loss
            loss = loss_fn(emotion_logits, emotions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = emotion_logits.argmax(dim=1)
            correct += (predictions == emotions).sum().item()
            total += emotions.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                acc = correct / total
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, Accuracy: {acc:.2%}")
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = correct / total
        print(f"[+] Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2%}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = f"weights/transformer_emotion_epoch{epoch+1}.pt"
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[+] Model saved to {save_path}")
    
    # Save final model
    save_path = "weights/transformer_emotion.pt"
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[+] Final model saved to {save_path}")


def main(args):
    """Main training function"""
    
    if args.model == 'detection':
        train_detection_model(args)
    elif args.model == 'emotion':
        train_emotion_model(args)
    elif args.model == 'both':
        print("[*] Training detection model...")
        train_detection_model(args)
        print("\n[*] Training emotion model...")
        train_emotion_model(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Models")
    parser.add_argument('--model', type=str, choices=['detection', 'emotion', 'both'], 
                       default='both', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--save-interval', type=int, default=5, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)
