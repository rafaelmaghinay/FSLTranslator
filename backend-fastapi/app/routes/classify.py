"""Classification endpoint."""

import base64
import traceback
from pathlib import Path
from typing import List

import numpy as np
import cv2
from fastapi import APIRouter, HTTPException

from app.core.config import TEMP_DIR, SEQ_LEN
from app.schemas.requests import ClassifyRequest
from app.schemas.responses import ClassifyResponse
from app.services.classify_service import preprocess_image, classify_sequence

router = APIRouter(prefix="/api", tags=["Classification"])


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify Filipino Sign Language gesture from cropped hand images.
    
    Accepts base64-encoded hand-cropped images and returns the predicted gesture
    with confidence scores using the BiLSTM neural network model.
    
    Args:
        request: ClassifyRequest containing list of base64 images
        
    Returns:
        Classification response with prediction and confidence
        
    Raises:
        HTTPException: If images are invalid or classification fails
    """
    cropped_images = request.cropped_images
    
    if not cropped_images:
        raise HTTPException(400, "No cropped images provided")
    
    print(f"🔍 Classifying {len(cropped_images)} base64 images...")
    
    # Convert base64 to temporary files
    temp_paths = []
    try:
        for i, base64_str in enumerate(cropped_images):
            try:
                # Extract base64 data
                if base64_str.startswith("data:image"):
                    base64_data = base64_str.split(",")[1]
                else:
                    base64_data = base64_str
                
                # Decode and save
                img_bytes = base64.b64decode(base64_data)
                temp_path = TEMP_DIR / f"classify_{i:03d}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)
                
                temp_paths.append(str(temp_path))
                print(f"   ✓ Converted image {i+1}")
                
            except Exception as e:
                print(f"   ❌ Error converting image {i+1}: {e}")
                raise HTTPException(400, f"Invalid base64 image at index {i}")
        
        # Load images and preprocess
        image_tensors = []
        for temp_path in temp_paths:
            img = cv2.imread(temp_path)
            if img is None:
                raise HTTPException(400, f"Failed to load image {temp_path}")
            
            tensor = preprocess_image(img)
            image_tensors.append(tensor)
        
        # Classify
        display_name, confidence, all_predictions = classify_sequence(image_tensors)
        
        return ClassifyResponse(
            ok=True,
            prediction=display_name,
            confidence=round(confidence, 4),
            all_predictions=all_predictions,
            frames_used=len(image_tensors)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Classification error: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Classification failed: {str(e)}")
