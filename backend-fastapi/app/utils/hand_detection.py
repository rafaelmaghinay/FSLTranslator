"""
Hand detection and image processing utilities.

Provides hand detection, cropping, and image preprocessing functions.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Optional, Tuple

from app.services.ml_model import model_manager
from app.core.config import HAND_DETECTION_PADDING


def detect_and_crop_single(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect hand in a single frame and return cropped region.
    
    Args:
        frame: Input frame (BGR format)
        
    Returns:
        Cropped hand region or None if no hand detected
    """
    detector = model_manager.get_yolo_detector()
    results = detector.predict(frame, conf=0.30, verbose=False)
    
    if not results or len(results) == 0 or results[0].boxes is None:
        return None
    
    h, w = frame.shape[:2]
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return None
    
    # Get best detection
    best_box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
    
    # Add padding
    x1 = max(0, x1 - HAND_DETECTION_PADDING)
    y1 = max(0, y1 - HAND_DETECTION_PADDING)
    x2 = min(w, x2 + HAND_DETECTION_PADDING)
    y2 = min(h, y2 + HAND_DETECTION_PADDING)
    
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def detect_and_crop_batch(frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    """
    Detect hands in multiple frames efficiently using batch processing.
    
    Args:
        frames: List of input frames (BGR format)
        
    Returns:
        List of cropped hand regions (None for frames with no detection)
    """
    detector = model_manager.get_yolo_detector()
    crops = []
    
    results = detector.predict(frames, conf=0.30, verbose=False)
    
    for i, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            crops.append(None)
            continue
        
        h, w = frames[i].shape[:2]
        best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
        
        # Add padding
        x1 = max(0, x1 - HAND_DETECTION_PADDING)
        y1 = max(0, y1 - HAND_DETECTION_PADDING)
        x2 = min(w, x2 + HAND_DETECTION_PADDING)
        y2 = min(h, y2 + HAND_DETECTION_PADDING)
        
        crop = frames[i][y1:y2, x1:x2]
        crops.append(crop if crop.size > 0 else None)
    
    return crops


def detect_boxes_yolo(frame: np.ndarray, conf_thres: float = 0.30, resize_input: bool = True) -> List[dict]:
    """
    Run YOLO detection and return bounding boxes.
    
    Optimized for both live detection (with resizing) and batch processing (without resizing).
    
    Args:
        frame: Input frame (BGR format)
        conf_thres: Confidence threshold
        resize_input: If True, pre-resize for faster inference
        
    Returns:
        List of detected boxes with coordinates
    """
    detector = model_manager.get_yolo_detector()
    h, w = frame.shape[:2]
    
    if resize_input:
        # For live: pre-resize to 256x256
        target_size = 256
        resized = cv2.resize(frame, (target_size, target_size))
        input_frame = resized
        scale = w / target_size
    else:
        # For batch: use original resolution
        input_frame = frame
        scale = 1.0
    
    results = detector.predict(
        input_frame,
        conf=conf_thres,
        imgsz=256,
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        half=torch.cuda.is_available(),
        max_det=2,
        agnostic_nms=True,
        classes=[0],  # Hand class
    )
    
    boxes = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            if resize_input:
                # Scale back to original
                x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            else:
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
