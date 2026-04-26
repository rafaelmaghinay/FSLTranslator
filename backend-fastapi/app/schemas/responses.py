"""Pydantic response models for API endpoints."""

from pydantic import BaseModel, Field
from typing import List, Optional, Any


class HealthResponse(BaseModel):
    """Health check response."""
    ok: bool
    message: str = "Backend is running"


class UploadSequenceResponse(BaseModel):
    """Response for image sequence upload endpoint."""
    ok: bool
    type: str = "sequence"
    uploaded: Optional[int] = None
    hands_detected: Optional[int] = None
    cropped_images: Optional[List[str]] = None
    error: Optional[str] = None


class UploadVideoResponse(BaseModel):
    """Response for video upload endpoint."""
    ok: bool
    type: str = "video"
    saved_as: Optional[str] = None
    frames_extracted: Optional[int] = None
    total_frames: Optional[int] = None
    cropped_images: Optional[List[str]] = None
    error: Optional[str] = None


class ClassifyResponse(BaseModel):
    """Response for gesture classification endpoint."""
    ok: bool
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    all_predictions: Optional[List[dict]] = None
    frames_used: Optional[int] = None
    error: Optional[str] = None


class ClearResponse(BaseModel):
    """Response for clear uploads endpoint."""
    ok: bool
    message: Optional[str] = None
    error: Optional[str] = None


class LiveFrameResponse(BaseModel):
    """Response for live frame reception endpoint."""
    ok: bool
    total_frames: int
    timestamp: float
    relative_time: float
    timestamp_source: str
    error: Optional[str] = None


class LiveStopResponse(BaseModel):
    """Response for live capture stop endpoint."""
    ok: bool
    total_frames: int
    hand_frames: int
    cropped_images: Optional[List[str]] = None
    timestamps: Optional[List[float]] = None
    relative_timestamps: Optional[List[float]] = None
    duration: Optional[float] = None
    avg_fps: Optional[float] = None
    sampling_strategy: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
