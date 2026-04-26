"""Pydantic request models for API endpoints."""

from pydantic import BaseModel, Field
from typing import List


class ClassifyRequest(BaseModel):
    """
    Request model for gesture classification from cropped hand images.
    
    Expects a sequence of base64-encoded hand-cropped images that will be
    classified using the BiLSTM neural network model.
    """
    
    cropped_images: List[str] = Field(
        ...,
        description="List of base64-encoded cropped hand images",
        example=["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."],
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "cropped_images": [
                    "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
                    "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
                ]
            }
        }
