"""
Machine learning models and inference engine.

Manages loading and inference of both the YOLO hand detection model
and the ResNet34+BiLSTM gesture classification model.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
from pathlib import Path
import warnings

from app.core.config import (
    BILSTM_MODEL_PATH, YOLO_MODEL_PATH, SEQ_LEN, IMG_SIZE,
    LSTM_HIDDEN, CLASSIFIER_DROPOUT, NUM_CLASSES
)

warnings.filterwarnings('ignore')


class ResNet34_BiLSTM(nn.Module):
    """
    Neural network combining ResNet34 feature extraction with bidirectional LSTM.
    
    Designed for Filipino Sign Language recognition from sequences of hand images.
    - ResNet34 extracts spatial features from each frame
    - BiLSTM models temporal dependencies across the sequence
    - Final classifier makes the gesture prediction
    
    Args:
        num_classes: Number of gesture classes to recognize
        lstm_hidden: Size of LSTM hidden state
        freeze_backbone: If True, freeze ResNet34 weights during inference
    """
    
    def __init__(self, num_classes: int, lstm_hidden: int = 96, freeze_backbone: bool = True):
        super().__init__()
        
        # Load pre-trained ResNet34
        weights = models.ResNet34_Weights.DEFAULT
        resnet = models.resnet34(weights=weights)
        self.feat_dim = 512
        
        # Remove classification head, keep feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Tensor of shape (batch, sequence_len, channels, height, width)
            
        Returns:
            Tensor of shape (batch, num_classes) with logits
        """
        # Flatten batch and sequence dimensions for ResNet
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        
        # Extract features
        feats = self.backbone(x_flat)
        
        # Reshape for LSTM
        feats = feats.view(B, S, -1)
        
        # Apply BiLSTM
        lstm_out, _ = self.lstm(feats)
        
        # Pool over time
        pooled = lstm_out.mean(dim=1)
        
        # Classify
        return self.classifier(pooled)


class ModelManager:
    """Manages loading and caching of ML models."""
    
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model manager and load models if not already loaded."""
        if not ModelManager._models_loaded:
            self._load_models()
            ModelManager._models_loaded = True
    
    def _load_models(self):
        """Load YOLO and BiLSTM models from disk."""
        print("🚀 Loading ML models...")
        
        # Load BiLSTM classifier
        self.classifier = ResNet34_BiLSTM(
            num_classes=NUM_CLASSES,
            lstm_hidden=LSTM_HIDDEN,
            freeze_backbone=True
        )
        
        try:
            checkpoint = torch.load(
                str(BILSTM_MODEL_PATH),
                map_location='cpu',
                weights_only=False
            )
            if 'model_state_dict' in checkpoint:
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.classifier.load_state_dict(checkpoint)
            print("✅ ResNet34+BiLSTM model loaded")
        except Exception as e:
            print(f"❌ Error loading BiLSTM model: {e}")
            raise
        
        self.classifier.eval()
        
        # Load YOLO hand detector
        try:
            self.yolo_detector = YOLO(str(YOLO_MODEL_PATH))
            if torch.cuda.is_available():
                self.yolo_detector.to('cuda')
                print("✅ YOLO detector loaded (GPU)")
            else:
                print("✅ YOLO detector loaded (CPU)")
        except Exception as e:
            print(f"❌ Error loading YOLO detector: {e}")
            raise
        
        print("✅ All models ready\n")
    
    def get_classifier(self):
        """Get the BiLSTM classifier model."""
        return self.classifier
    
    def get_yolo_detector(self):
        """Get the YOLO hand detector model."""
        return self.yolo_detector


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# Global model manager instance
model_manager = ModelManager()
