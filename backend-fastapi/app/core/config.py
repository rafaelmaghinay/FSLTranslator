"""
Application configuration and environment variables.

Centralized configuration management for the FSL Translator backend.
Supports both local development and production deployments.
"""

import os
from pathlib import Path
from typing import List

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
IMG_DIR = UPLOADS_DIR / "images"
SEQ_DIR = UPLOADS_DIR / "sequences"
VID_DIR = UPLOADS_DIR / "videos"
TEMP_DIR = UPLOADS_DIR / "temp"

# API Configuration
API_TITLE = "FSL Translator Backend"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Filipino Sign Language Recognition API"

# CORS Configuration
ALLOWED_ORIGINS: List[str] = [
    "http://localhost:5173",  # Local development
    "http://localhost:3000",  # Alternative dev port
    "https://fslrecognizer.vercel.app",  # Production
    "https://fsl-translator-api.chickenkiller.com",  # Custom domain
]

# Model Configuration
BILSTM_MODEL_PATH = MODELS_DIR / "bilstm_best_test14.pth"
YOLO_MODEL_PATH = MODELS_DIR / "yolo_best.pt"

# ML Model Parameters
SEQ_LEN = 20  # Sequence length for LSTM
IMG_SIZE = 224  # Input image size for ResNet
LSTM_HIDDEN = 128  # LSTM hidden state size
CLASSIFIER_DROPOUT = 0.4  # Dropout rate for classifier

NUM_CLASSES = 43  # Total number of sign language classes

# Video Processing
TARGET_VIDEO_FRAMES = 20
YOLO_BATCH_SIZE = 8
YOLO_CONFIDENCE_THRESHOLD = 0.30
HAND_DETECTION_PADDING = 20

# Frame skipping for live detection
FRAME_SKIP = 2

# WebSocket Configuration
WEBSOCKET_FRAME_SKIP = 3

# Ensure directories exist
for directory in [IMG_DIR, SEQ_DIR, VID_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
