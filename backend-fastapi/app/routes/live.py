"""Live detection and webcam endpoints."""

import time
import json
import base64
import gc
import traceback
from typing import Optional

import numpy as np
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Form, HTTPException

from app.core.config import SEQ_LEN, TEMP_DIR
from app.schemas.responses import LiveFrameResponse, LiveStopResponse
from app.services.webcam_service import get_or_create_session, get_session, stop_session, active_sessions
from app.utils.hand_detection import detect_boxes_yolo

router = APIRouter(prefix="/api/live", tags=["Live Detection"])


@router.websocket("/ws")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint for real-time hand detection visualization.
    
    Accepts base64-encoded video frames and returns bounding boxes
    for detected hands with minimal latency.
    """
    await websocket.accept()
    print("🔌 WebSocket connected for live detection")
    
    last_boxes = []
    frame_skip_counter = 0
    FRAME_SKIP = 3
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                frame_b64 = message.get("frame")
                
                if not frame_b64:
                    await websocket.send_json({"error": "No frame provided"})
                    continue
                
                # Frame skipping for performance
                frame_skip_counter += 1
                if frame_skip_counter % FRAME_SKIP != 0:
                    await websocket.send_json({
                        "boxes": last_boxes,
                        "count": len(last_boxes),
                        "timestamp": time.time(),
                        "cached": True
                    })
                    continue
                
                # Decode frame
                if frame_b64.startswith("data:image"):
                    frame_b64 = frame_b64.split(",")[1]
                
                frame_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Invalid frame"})
                    continue
                
                # Detect boxes
                boxes = detect_boxes_yolo(frame, conf_thres=0.25, resize_input=True)
                last_boxes = boxes
                
                # Send response
                await websocket.send_json({
                    "boxes": boxes,
                    "count": len(boxes),
                    "timestamp": time.time(),
                    "cached": False
                })
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        traceback.print_exc()


@router.post("/capture")
async def start_capture(session_id: Optional[str] = Form(None)):
    """
    Start a new capture session for recording webcam frames.
    
    Args:
        session_id: Optional session ID; creates new if not provided
        
    Returns:
        Session information with ID and status
        
    Raises:
        HTTPException: If session creation fails
    """
    try:
        session = get_or_create_session(session_id)
        session_id = session["id"]
        
        # Initialize capture
        session["is_capturing"] = True
        session["start_time"] = time.time()
        session["frames"] = []
        session["timestamps"] = []
        session["hand_crops"] = []
        
        print(f"🎬 Started capture session: {session_id}")
        
        return {
            "ok": True,
            "session_id": session_id,
            "message": "Capture started. Send frames to /api/live/frame"
        }
    
    except Exception as e:
        print(f"❌ Error starting capture: {e}")
        raise HTTPException(500, f"Failed to start capture: {str(e)}")


@router.post("/frame", response_model=LiveFrameResponse)
async def receive_frame(
    session_id: str = Form(...),
    frame: str = Form(...),
    timestamp: Optional[float] = Form(None)
) -> LiveFrameResponse:
    """
    Store a video frame for the capture session.
    
    Receives base64-encoded frames and stores them with timestamps
    for later batch processing and hand detection.
    
    Args:
        session_id: Session identifier
        frame: Base64-encoded frame
        timestamp: Optional client-side timestamp in milliseconds
        
    Returns:
        Confirmation with frame count and timestamp info
        
    Raises:
        HTTPException: If session not found or frame invalid
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(404, "Session not found")
        
        session = active_sessions[session_id]
        
        if not session.get("is_capturing"):
            raise HTTPException(400, "Session is not capturing")
        
        # Decode frame
        if frame.startswith("data:image"):
            frame = frame.split(",")[1]
        
        frame_bytes = base64.b64decode(frame)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame_bgr is None:
            raise HTTPException(400, "Invalid frame")
        
        # Handle timestamps
        if timestamp is not None:
            client_timestamp = timestamp / 1000.0
            if session.get("start_time") is None:
                session["start_time"] = client_timestamp
            relative_timestamp = client_timestamp - session["start_time"]
            used_timestamp = client_timestamp
            timestamp_source = "client"
        else:
            server_timestamp = time.time()
            if session.get("start_time") is None:
                session["start_time"] = server_timestamp
            relative_timestamp = server_timestamp - session["start_time"]
            used_timestamp = server_timestamp
            timestamp_source = "server"
        
        # Store frame
        session["frames"].append(frame_bgr.copy())
        session["timestamps"].append(used_timestamp)
        
        return LiveFrameResponse(
            ok=True,
            total_frames=len(session["frames"]),
            timestamp=used_timestamp,
            relative_time=round(relative_timestamp, 3),
            timestamp_source=timestamp_source
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error receiving frame: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Failed to process frame: {str(e)}")


@router.post("/stop", response_model=LiveStopResponse)
async def stop_capture(session_id: str = Form(...)) -> LiveStopResponse:
    """
    Stop capture session and process frames for gesture recognition.
    
    Processes all captured frames with hand detection and applies
    10/80/10 temporal sampling strategy to prepare for classification.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Processed hand crops as base64 images
        
    Raises:
        HTTPException: If session not found or processing fails
    """
    try:
        if session_id not in active_sessions:
            raise HTTPException(404, "Session not found")
        
        session = active_sessions[session_id]
        
        if not session.get("is_capturing"):
            raise HTTPException(400, "Session is not capturing")
        
        # Stop capturing
        session["is_capturing"] = False
        duration = time.time() - session.get("start_time", 0)
        
        frames = session["frames"]
        timestamps = session.get("timestamps", [])
        total_frames = len(frames)
        
        print(f"⏹️ Stopped capture session: {session_id}")
        print(f"   Duration: {duration:.2f}s, Frames: {total_frames}")
        
        if total_frames == 0:
            return LiveStopResponse(
                ok=False,
                total_frames=0,
                hand_frames=0,
                error="No frames captured"
            )
        
        # Detect hands in all frames
        from app.utils.hand_detection import detect_and_crop_batch
        
        print(f"🔍 Processing {total_frames} frames with hand detection...")
        crops = detect_and_crop_batch(frames)
        valid_crops = [(crop, i, ts) for i, (crop, ts) in enumerate(zip(crops, timestamps))
                       if crop is not None and crop.size > 0]
        
        if not valid_crops:
            return LiveStopResponse(
                ok=False,
                total_frames=total_frames,
                hand_frames=0,
                error="No hands detected"
            )
        
        # Apply 10/80/10 sampling
        selected_crops = _apply_temporal_sampling([c[0] for c in valid_crops])
        
        # Convert to base64
        cropped_b64 = []
        for crop in selected_crops:
            _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buffer).decode()
            cropped_b64.append(f"data:image/jpeg;base64,{crop_b64}")
        
        # Cleanup
        session["frames"] = []
        session["timestamps"] = []
        gc.collect()
        
        return LiveStopResponse(
            ok=True,
            total_frames=total_frames,
            hand_frames=len(cropped_b64),
            cropped_images=cropped_b64,
            duration=round(duration, 2),
            sampling_strategy="10/80/10",
            message=f"Captured {total_frames} frames, detected {len(cropped_b64)} hands"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error stopping capture: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Failed to stop capture: {str(e)}")


def _apply_temporal_sampling(crops: list) -> list:
    """
    Apply 10/80/10 temporal sampling strategy.
    
    If insufficient crops: pad by duplicating middle frames
    If sufficient crops: uniformly sample 10% from beginning, 80% from middle, 10% from end
    """
    target_len = SEQ_LEN
    
    if len(crops) < target_len:
        # Pad strategy
        result = crops.copy()
        mid_idx = len(crops) // 2
        while len(result) < target_len:
            result.insert(mid_idx + 1, crops[mid_idx].copy())
        return result
    
    else:
        # Sample strategy
        total = len(crops)
        begin_end = max(1, int(total * 0.10))
        middle_start = begin_end
        middle_end = max(middle_start + 1, int(total * 0.90))
        
        # Sample from each region
        begin = crops[:begin_end]
        middle = crops[middle_start:middle_end]
        end = crops[middle_end:]
        
        num_begin = int(target_len * 0.10)
        num_middle = int(target_len * 0.80)
        num_end = target_len - num_begin - num_middle
        
        def sample_region(region, num_samples):
            if len(region) <= num_samples:
                return region
            indices = np.linspace(0, len(region) - 1, num_samples, dtype=int)
            return [region[i] for i in indices]
        
        sampled = (sample_region(begin, num_begin) +
                   sample_region(middle, num_middle) +
                   sample_region(end, num_end))
        
        return sampled[:target_len]
