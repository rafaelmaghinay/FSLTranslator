"""
File handling utilities for upload processing.

Provides functions for file I/O, cleanup, and directory management.
"""

import os
import gc
import shutil
from pathlib import Path
from fastapi import UploadFile


def save_file(upload: UploadFile, destination: Path) -> None:
    """
    Save uploaded file to disk with automatic directory creation.
    
    Args:
        upload: FastAPI UploadFile object
        destination: Target file path
        
    Raises:
        OSError: If file cannot be written
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def clear_directory(directory: Path) -> bool:
    """
    Delete all files and subdirectories, then recreate folder structure.
    
    Args:
        directory: Path to directory to clear
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Force garbage collection to release file handles
        gc.collect()
        
        print(f"🗑️  Clearing {directory}...")
        
        if directory.exists():
            # Remove all files and subdirectories
            for item in directory.iterdir():
                try:
                    if item.is_file():
                        os.remove(str(item))
                        print(f"   ✓ Deleted file: {item.name}")
                    elif item.is_dir():
                        file_count = len(list(item.rglob('*')))
                        shutil.rmtree(str(item))
                        print(f"   ✓ Deleted dir: {item.name} ({file_count} files)")
                except Exception as e:
                    print(f"   ⚠ Error deleting {item.name}: {e}")
        
        # Recreate subdirectories
        subdirs = ['images', 'sequences', 'videos', 'temp']
        for subdir_name in subdirs:
            subdir = directory / subdir_name
            subdir.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Recreated dir: {subdir_name}")
        
        print("✅ Clear completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing uploads: {e}")
        return False
