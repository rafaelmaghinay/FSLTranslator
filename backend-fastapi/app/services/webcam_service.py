"""
Webcam and live detection session management.

Manages active capture sessions for live hand detection and recording.
"""

import uuid
import time
from typing import Dict, Optional, List
import cv2
import numpy as np

# Active capture sessions
active_sessions: Dict[str, Dict] = {}


def get_or_create_session(session_id: Optional[str] = None) -> Dict:
    """
    Get existing session or create a new one.
    
    Args:
        session_id: Optional session ID; creates new if not provided
        
    Returns:
        Session dictionary
    """
    if session_id and session_id in active_sessions:
        return active_sessions[session_id]
    
    new_id = session_id or str(uuid.uuid4())
    active_sessions[new_id] = {
        "id": new_id,
        "is_capturing": False,
        "frames": [],
        "timestamps": [],
        "hand_crops": [],
        "start_time": None
    }
    return active_sessions[new_id]


def stop_session(session_id: str) -> bool:
    """
    Terminate and cleanup session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if successful
    """
    if session_id in active_sessions:
        session = active_sessions[session_id]
        session["is_capturing"] = False
        session["frames"] = []
        session["timestamps"] = []
        session["hand_crops"] = []
        return True
    return False


def get_session(session_id: str) -> Optional[Dict]:
    """
    Retrieve active session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session dictionary or None if not found
    """
    return active_sessions.get(session_id)
