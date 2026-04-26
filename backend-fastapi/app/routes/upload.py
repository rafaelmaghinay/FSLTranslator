"""Upload and preprocessing endpoints."""

import time
import traceback
import base64
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.core.config import IMG_DIR, SEQ_DIR, VID_DIR, TEMP_DIR
from app.schemas.responses import UploadSequenceResponse, UploadVideoResponse
from app.utils.file_utils import save_file
from app.utils.hand_detection import detect_and_crop_batch, detect_boxes_yolo

router = APIRouter(prefix="/api/upload", tags=["Upload"])


@router.post("/sequence", response_model=UploadSequenceResponse)
async def upload_sequence(images: List[UploadFile] = File(...)) -> UploadSequenceResponse:
    """
    Process image sequence for sign recognition.
    
    Detects hands in each uploaded image and returns base64-encoded cropped regions.
    
    Args:
        images: List of image files to process
        
    Returns:
        Response with cropped hand images as base64 strings
        
    Raises:
        HTTPException: If no images provided or files are not valid images
    """
    if not images:
        raise HTTPException(400, "No images uploaded")

    # Validate and save images
    saved_paths = []
    for img in images:
        if not img.content_type or not img.content_type.startswith("image/"):
            raise HTTPException(400, "All files must be images")

        ext = Path(img.filename).suffix
        filename = f"{int(time.time()*1000)}_{Path(img.filename).stem}{ext}"
        path = SEQ_DIR / filename
        save_file(img, path)
        saved_paths.append(str(path))
    
    try:
        # Load all images
        frames = []
        for path in saved_paths:
            frame = cv2.imread(path)
            if frame is not None:
                frames.append(frame)
        
        if not frames:
            raise HTTPException(400, "Could not load any images")
        
        # Detect hands
        crops = detect_and_crop_batch(frames)
        valid_crops = [c for c in crops if c is not None and c.size > 0]
        
        if len(valid_crops) < 8:
            return UploadSequenceResponse(
                ok=False,
                error=f"Only {len(valid_crops)} hands detected. Need at least 8."
            )
        
        # Convert to base64
        cropped_base64 = []
        for crop in valid_crops:
            _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buffer).decode()
            cropped_base64.append(f"data:image/jpeg;base64,{crop_b64}")
        
        return UploadSequenceResponse(
            ok=True,
            type="sequence",
            uploaded=len(saved_paths),
            hands_detected=len(valid_crops),
            cropped_images=cropped_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing sequence: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Failed to process sequence: {str(e)}")


@router.post("/video", response_model=UploadVideoResponse)
async def upload_video(video: UploadFile = File(...)) -> UploadVideoResponse:
    """
    Extract frames from video and detect hand regions.
    
    Args:
        video: Video file to process
        
    Returns:
        Response with cropped hand images from video frames
        
    Raises:
        HTTPException: If file is not a valid video or processing fails
    """
    if not video.content_type or "video" not in video.content_type:
        raise HTTPException(400, "Only video files allowed")

    # Save video
    ext = Path(video.filename).suffix or ".mp4"
    filename = f"{int(time.time()*1000)}_{Path(video.filename).stem}{ext}"
    path = VID_DIR / filename
    save_file(video, path)

    try:
        # Extract frames from video
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise HTTPException(400, "Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames == 0:
            cap.release()
            raise HTTPException(400, "Video has 0 frames")
        
        # Sample frames uniformly
        frames = []
        sample_indices = np.linspace(0, total_frames - 1, min(total_frames, 60)).astype(int)
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise HTTPException(400, "Could not extract any frames from video")
        
        # Detect hands
        crops = detect_and_crop_batch(frames)
        valid_crops = [c for c in crops if c is not None and c.size > 0]
        
        if not valid_crops:
            raise HTTPException(400, "No hands detected in any frame")
        
        # Convert to base64
        cropped_b64 = []
        for crop in valid_crops:
            _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buffer).decode()
            cropped_b64.append(f"data:image/jpeg;base64,{crop_b64}")
        
        return UploadVideoResponse(
            ok=True,
            type="video",
            saved_as=filename,
            frames_extracted=len(frames),
            total_frames=total_frames,
            cropped_images=cropped_b64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Failed to process video: {str(e)}")
