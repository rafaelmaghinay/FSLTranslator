"""
Gesture classification service.

Handles inference using the BiLSTM model and prediction postprocessing.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image
import cv2

from app.services.ml_model import model_manager, transform
from app.core.constants import CLASS_NAMES, clean_class_name


def preprocess_image(image_data) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image_data: PIL Image or numpy array (BGR)
        
    Returns:
        Normalized tensor ready for model input
    """
    if isinstance(image_data, np.ndarray):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = Image.fromarray(rgb)
    
    return transform(image_data)


def classify_sequence(image_tensors: List[torch.Tensor]) -> Tuple[str, float, List[dict]]:
    """
    Classify gesture from sequence of hand-cropped images.
    
    Args:
        image_tensors: List of preprocessed image tensors
        
    Returns:
        Tuple of (predicted_class_name, confidence, all_predictions)
    """
    classifier = model_manager.get_classifier()
    
    # Stack into batch
    sequence = torch.stack(image_tensors).unsqueeze(0)  # (1, seq_len, 3, 224, 224)
    
    with torch.no_grad():
        logits = classifier(sequence)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
    
    pred_idx = int(pred_idx.item())
    confidence = float(conf.item())
    
    predicted_class = CLASS_NAMES[pred_idx]
    display_name = clean_class_name(predicted_class)
    
    # Get all predictions sorted by confidence
    all_conf, all_idx = torch.topk(probs[0], len(CLASS_NAMES))
    all_predictions = [
        {
            "class": CLASS_NAMES[int(idx)],
            "label": clean_class_name(CLASS_NAMES[int(idx)]),
            "confidence": float(conf)
        }
        for conf, idx in zip(all_conf, all_idx)
    ]
    
    return display_name, confidence, all_predictions
