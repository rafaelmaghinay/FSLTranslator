"""Utility and management endpoints."""

import traceback
from fastapi import APIRouter, HTTPException

from app.core.config import UPLOADS_DIR
from app.utils.file_utils import clear_directory
from app.schemas.responses import ClearResponse

router = APIRouter(prefix="/api", tags=["Utilities"])


@router.post("/clear", response_model=ClearResponse)
async def clear_all() -> ClearResponse:
    """
    Delete all stored uploads and reset upload directories.
    
    This endpoint clears all image sequences, videos, and temporary files
    while recreating the necessary directory structure.
    
    Returns:
        Confirmation message
        
    Raises:
        HTTPException: If cleanup fails
    """
    try:
        success = clear_directory(UPLOADS_DIR)
        if success:
            return ClearResponse(ok=True, message="All uploads cleared")
        else:
            raise HTTPException(500, "Failed to clear uploads")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, f"Error clearing uploads: {str(e)}")
