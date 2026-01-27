import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import time
import json
import uuid
from datetime import datetime
import os
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo_best.pt"
UPLOADS_DIR = BASE_DIR / "uploads"  
TEMP_DIR = UPLOADS_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 20 
IMG_SIZE = 224

CONFIDENCE_THRESHOLD = 0.5

active_sessions = {}

YOLO_IMGSZ = 256  # ⬅️ REDUCED from 224 to 320 (faster inference)
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"
YOLO_HALF = bool(torch.cuda.is_available())  # FP16 for GPU

# ⬅️ ADD THIS: Initialize YOLO detector
print("🔍 Loading YOLO hand detector for webcam...")
detector = YOLO(str(YOLO_MODEL_PATH))
if torch.cuda.is_available():
    detector.to('cuda')
    print("✅ YOLO running on GPU")
else:
    print("⚠️ YOLO running on CPU (slower)")

# In webcam.py, update yolo_detect_boxes function:

def yolo_detect_boxes(frame, conf_thres=0.30, resize_input=True):
    """
    Run YOLO detection with optimized settings.
    
    Args:
        frame: Input frame (BGR)
        conf_thres: Confidence threshold
        resize_input: If True, resize frame to 256x256 before YOLO.
                     If False, let YOLO handle resizing internally.
    
    Returns:
        list: Detected boxes with coordinates
    """
    h, w = frame.shape[:2]
    
    if resize_input:
        # ⬅️ FOR LIVE DETECTION: Pre-resize to 256x256 for speed
        target_size = 256
        resized = cv2.resize(frame, (target_size, target_size))
        input_frame = resized
        scale = w / target_size
    else:
        # ⬅️ FOR CAPTURE/STOP: Use original frame, let YOLO resize internally
        input_frame = frame
        scale = 1.0  # No scaling needed
    
    # Run YOLO with optimized settings
    results = detector.predict(
        input_frame,
        conf=conf_thres,
        imgsz=256,  # YOLO internal resize target
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        half=torch.cuda.is_available(),
        max_det=2,  
        agnostic_nms=True,
        classes=[0]  # Only detect class 0 (hand)
    )
    
    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            if resize_input:
                # Scale back to original frame size
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            else:
                # Already in original frame coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clamp to frame bounds
            x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            
            boxes.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": conf
            })
    
    return boxes



def get_or_create_session(session_id=None):
    """Get existing session or create new one."""
    if session_id and session_id in active_sessions:
        return active_sessions[session_id]
    
    new_id = session_id or str(uuid.uuid4())
    active_sessions[new_id] = {
        "id": new_id,
        "is_capturing": False,
        "frames": [],
        "hand_crops": [],
        "start_time": None
    }
    return active_sessions[new_id]

def stop_session(session_id):
    """Stop and cleanup session."""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        session["is_capturing"] = False
        session["frames"] = []
        session["hand_crops"] = []
        return True
    return False

def capture_session_video(session_id):
    """Get captured frames from session."""
    if session_id in active_sessions:
        return active_sessions[session_id]
    return None