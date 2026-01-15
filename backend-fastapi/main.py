from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
from pathlib import Path
import shutil
import time
import base64
import gc
import os
import time as time_module
import traceback
import numpy as np
import cv2
from pydantic import BaseModel
import io
from fastapi import Form
import asyncio
import json
from upload import process_single_image, process_image_sequence, process_video, classify_cropped_images
from webcam import yolo_detect_boxes, get_or_create_session, active_sessions, SEQ_LEN, stop_session, capture_session_video


app = FastAPI(title="FSL Upload Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS = BASE_DIR / "uploads"
IMG_DIR = UPLOADS / "images"
SEQ_DIR = UPLOADS / "sequences"
VID_DIR = UPLOADS / "videos"
TEMP_DIR = UPLOADS / "temp"

frame_skip_counter = 0
FRAME_SKIP = 2  

class ClassifyRequest(BaseModel):
    cropped_images: List[str]  # base64 strings

for d in [IMG_DIR, SEQ_DIR, VID_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ⬅️ MOUNT STATIC FILES - Add this BEFORE defining routes
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")

def save_file(upload: UploadFile, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

def clear_uploads(directory: Path):
    """Clear all files in uploads directory while preserving structure."""
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
                        shutil.rmtree(str(item))
                        print(f"   ✓ Deleted dir: {item.name}")
                except Exception as e:
                    print(f"   ⚠ Error deleting {item.name}: {e}")
        
        # Recreate subdirectories
        for subdir in [IMG_DIR, SEQ_DIR, VID_DIR, TEMP_DIR]:
            subdir.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Created dir: {subdir.name}")
        
        print("✅ Clear completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error clearing uploads: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.get("/")
def health():
    return {"ok": True, "message": "Backend is running"}

# ============================================================
# UPLOAD & CROP ENDPOINTS
# ============================================================

@app.post("/api/upload/image")
async def upload_image(image: UploadFile = File(...)):
    """Upload single image, detect hand, return cropped image as base64."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files allowed")

    ext = Path(image.filename).suffix
    filename = f"{int(time.time()*1000)}_{Path(image.filename).stem}{ext}"
    path = IMG_DIR / filename

    save_file(image, path)
    
    # Process: detect hand and crop
    result = process_single_image(str(path))
    
    if result["status"] == "success":
        # Convert cropped image to base64
        with open(result["cropped_image_path"], "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
            cropped_base64 = f"data:image/jpeg;base64,{img_data}"
        
        return {
            "ok": True,
            "type": "image",
            "saved_as": filename,
            "cropped_image": cropped_base64  # ⬅️ Base64 string
        }
    elif result["status"] == "no_hand_detected":
        return {
            "ok": False,
            "type": "image",
            "error": "No hand detected in image"
        }
    else:
        raise HTTPException(500, result.get("error", "Unknown error"))

@app.post("/api/upload/sequence")
async def upload_sequence(images: List[UploadFile] = File(...)):
    """Upload multiple images, detect hands in all, return cropped images as base64."""
    if not images:
        raise HTTPException(400, "No images uploaded")

    saved_paths = []
    for img in images:
        if not img.content_type or not img.content_type.startswith("image/"):
            raise HTTPException(400, "All files must be images")

        ext = Path(img.filename).suffix
        filename = f"{int(time.time()*1000)}_{Path(img.filename).stem}{ext}"
        path = SEQ_DIR / filename
        save_file(img, path)
        saved_paths.append(str(path))
    
    # Process: detect hands in all and crop
    result = process_image_sequence(saved_paths)
    
    if result["status"] == "success":
        # Convert cropped images to base64
        cropped_base64 = []
        for crop_path in result["cropped_images"]:
            with open(crop_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
                cropped_base64.append(f"data:image/jpeg;base64,{img_data}")
        
        return {
            "ok": True,
            "type": "sequence",
            "uploaded": len(saved_paths),
            "hands_detected": result["count"],
            "cropped_images": cropped_base64  # ⬅️ Only base64 strings
        }
    elif result["status"] == "insufficient_hands":
        return {
            "ok": False,
            "type": "sequence",
            "error": f"Only {result['count']} hands detected. Need at least {result['required']}"
        }
    else:
        raise HTTPException(500, result.get("error", "Unknown error"))

@app.post("/api/upload/video")
async def upload_video(video: UploadFile = File(...)):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(400, "Only video files allowed")

    ext = Path(video.filename).suffix or ".mp4"
    filename = f"{int(time.time()*1000)}_{Path(video.filename).stem}{ext}"
    path = VID_DIR / filename
    save_file(video, path)

    try:
        result = process_video(str(path))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"process_video crashed: {e}")

    if result.get("status") != "success":
        # Return the real error instead of a silent 500
        raise HTTPException(500, result.get("error", "process_video failed"))

    cropped_b64 = []
    for crop_path in result["cropped_images"]:
        with open(crop_path, "rb") as f:
            cropped_b64.append(
                "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
            )

    return {
        "ok": True,
        "type": "video",
        "saved_as": filename,
        "frames_extracted": result["count"],
        "total_frames": result["total_frames"],
        "cropped_images": cropped_b64,
    }
    


@app.post("/api/classify")
async def classify(request: ClassifyRequest):
    """
    Classify a sequence of cropped images from base64 strings.
    
    Expected input:
    {
        "cropped_images": ["data:image/jpeg;base64,/9j/4AAQSk...", ...]
    }
    
    Returns top 3 predictions with confidence scores.
    """
    cropped_images = request.cropped_images
    
    if not cropped_images:
        raise HTTPException(400, "No cropped images provided")
    
    print(f"🔍 Classifying {len(cropped_images)} base64 images...")
    
    # Convert base64 strings to temporary files
    temp_paths = []
    try:
        for i, base64_str in enumerate(cropped_images):
            try:
                # Extract base64 data (remove "data:image/jpeg;base64," prefix)
                if base64_str.startswith("data:image"):
                    base64_data = base64_str.split(",")[1]
                else:
                    base64_data = base64_str
                
                # Decode base64 to bytes
                img_bytes = base64.b64decode(base64_data)
                
                # Save to temporary file
                temp_path = TEMP_DIR / f"classify_{i:03d}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)
                
                temp_paths.append(str(temp_path))
                print(f"   ✓ Converted base64 image {i+1}")
            except Exception as e:
                print(f"   ❌ Error converting image {i+1}: {e}")
                raise HTTPException(400, f"Invalid base64 image at index {i}")
        
        # Classify using the temporary file paths - NOW RETURNS TOP 3
        result = classify_cropped_images(temp_paths)
        
        if result["status"] == "success":
            return {
                "ok": True,
                "prediction": result["prediction"],
                "confidence": round(result["confidence"], 4),
                "top_3": result["top_3"],  # ⬅️ NEW: Top 3 results
                "frames_used": result["frames_used"]
            }
        else:
            raise HTTPException(500, result.get("error", "Classification failed"))
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Classification error: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Classification failed: {str(e)}")



# ============================================================
# CLEAR ENDPOINT
# ============================================================

@app.post("/api/clear")
async def clear_all():
    """Clear all uploaded files."""
    try:
        success = clear_uploads(UPLOADS)
        if success:
            return {"ok": True, "message": "All uploads cleared"}
        else:
            raise HTTPException(500, "Failed to clear uploads")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, f"Error clearing uploads: {str(e)}")
    


@app.websocket("/api/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for live hand detection."""
    await websocket.accept()
    print("🔌 WebSocket connected for live detection")
    
    frame_skip_counter = 0
    last_boxes = []
    FRAME_SKIP = 3  # ⬅️ INCREASE from 2 to 3 (process every 3rd frame)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                frame_b64 = message.get("frame")
                
                if not frame_b64:
                    await websocket.send_json({"error": "No frame provided"})
                    continue
                
                # Frame skipping: only process every 3rd frame
                frame_skip_counter += 1
                if frame_skip_counter % FRAME_SKIP != 0:
                    # Return cached result immediately
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
                
                # Detect hands
                boxes = yolo_detect_boxes(frame, conf_thres=0.25)  # ⬅️ Lower threshold
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



@app.post("/api/live/capture")
async def start_capture(session_id: str = Form(None)):
    """
    Start capturing video from live webcam.
    Creates a session to store frames with detected hands.
    
    Returns:
        dict: {"ok": True, "session_id": str, "message": str}
    """
    try:
        # Create or get session
        session = get_or_create_session(session_id)
        session_id = session["id"]
        
        # Start capturing
        session["is_capturing"] = True
        session["start_time"] = time.time()
        session["frames"] = []
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


@app.post("/api/live/frame")
async def receive_frame(
    session_id: str = Form(...),
    frame: str = Form(...)
):
    """
    OPTIMIZED: Just store frames, defer YOLO detection to /api/live/stop.
    This makes capture 5-10x faster!
    
    Args:
        session_id: Session identifier
        frame: Base64 encoded frame
    
    Returns:
        dict: {"ok": True, "total_frames": int}
    """
    try:
        # Get session
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
        
        # ⬅️ OPTIMIZATION: Just store the frame, don't detect yet!
        session["frames"].append(frame_bgr.copy())
        
        # Return immediately (much faster!)
        return {
            "ok": True,
            "total_frames": len(session["frames"])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error receiving frame: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Failed to process frame: {str(e)}")


@app.post("/api/live/stop")
async def stop_capture(session_id: str = Form(...)):
    """
    OPTIMIZED: Stop capture, then batch-process all frames with YOLO.
    Extract exactly SEQ_LEN hand crops IN SEQUENCE for classification.
    NOW DETECTS BOTH HANDS (left and right) if present.
    
    Args:
        session_id: Session identifier
    
    Returns:
        dict: {
            "ok": True,
            "total_frames": int,
            "hand_frames": int,
            "cropped_images": List[str],  # base64 encoded crops
            "message": str
        }
    """
    try:
        # Get session
        if session_id not in active_sessions:
            raise HTTPException(404, "Session not found")
        
        session = active_sessions[session_id]
        
        if not session.get("is_capturing"):
            raise HTTPException(400, "Session is not capturing")
        
        # Stop capturing
        session["is_capturing"] = False
        duration = time.time() - session.get("start_time", 0)
        
        frames = session["frames"]
        total_frames = len(frames)
        
        print(f"⏹️ Stopped capture session: {session_id}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Total frames captured: {total_frames}")
        
        if total_frames == 0:
            return {
                "ok": False,
                "error": "No frames captured",
                "total_frames": 0,
                "hand_frames": 0
            }
        
        # ⬅️ PROCESS ALL FRAMES IN SEQUENCE - DETECT BOTH HANDS
        print(f"🔍 Processing {total_frames} frames with YOLO (detecting both hands)...")
        
        all_hand_crops = []
        frame_indices = []
        
        for idx, frame in enumerate(frames):
            # Detect ALL hands in this frame (not just first one!)
            boxes = yolo_detect_boxes(frame, conf_thres=0.30)
            
            if len(boxes) == 0:
                continue
            
            h, w = frame.shape[:2]
            
            # ⬅️ CRITICAL FIX: Process ALL detected hands, not just first one
            # Sort boxes left-to-right to maintain consistency (left hand, then right hand)
            boxes_sorted = sorted(boxes, key=lambda b: b["x1"])
            
            frame_crops = []
            for box in boxes_sorted:
                x1 = int(box["x1"])
                y1 = int(box["y1"])
                x2 = int(box["x2"])
                y2 = int(box["y2"])
                
                # Add padding
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    frame_crops.append(crop.copy())
            
            # Add all crops from this frame
            if len(frame_crops) > 0:
                all_hand_crops.extend(frame_crops)
                # Track frame index for each crop
                frame_indices.extend([idx] * len(frame_crops))
        
        print(f"   ✅ Detected {len(all_hand_crops)} hand crops across {len(set(frame_indices))} frames")
        print(f"   Hands per frame: {len(all_hand_crops) / max(len(set(frame_indices)), 1):.1f} average")
        print(f"   Frame indices with hands: {sorted(set(frame_indices))[:10]}..." if len(set(frame_indices)) > 10 else f"   Frame indices: {sorted(set(frame_indices))}")
        
        # ⬅️ Sample uniformly while preserving temporal order
        if len(all_hand_crops) < SEQ_LEN:
            if len(all_hand_crops) > 0:
                # Pad with last frame (maintains sequence)
                original_count = len(all_hand_crops)
                while len(all_hand_crops) < SEQ_LEN:
                    all_hand_crops.append(all_hand_crops[-1])
                message = f"⚠️ Only {original_count} hand crops detected. Padded to {SEQ_LEN} by repeating last crop."
            else:
                return {
                    "ok": False,
                    "error": "No hands detected in any frame",
                    "total_frames": total_frames,
                    "hand_frames": 0
                }
        elif len(all_hand_crops) > SEQ_LEN:
            # Sample uniformly to get exactly SEQ_LEN crops IN SEQUENCE
            indices = np.linspace(0, len(all_hand_crops) - 1, SEQ_LEN, dtype=int)
            
            sampled_crops = [all_hand_crops[i] for i in indices]
            sampled_indices = [frame_indices[i] for i in indices]
            
            print(f"   📊 Sampled crop indices: {sampled_indices}")
            
            all_hand_crops = sampled_crops
            message = f"✅ Sampled {SEQ_LEN} crops from {len(frame_indices)} detected (in sequence, both hands included)"
        else:
            message = f"✅ Perfect! Got exactly {SEQ_LEN} hand crops"
        
        # Convert crops to base64 (in sequence!)
        cropped_b64 = []
        for i, crop in enumerate(all_hand_crops):
            # Resize to 640x640 (YOLO training size) for consistency
            crop_resized = cv2.resize(crop, (640, 640))
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buffer).decode()
            cropped_b64.append(f"data:image/jpeg;base64,{crop_b64}")
        
        print(f"   {message}")
        print(f"   Returning {len(cropped_b64)} cropped hand images (640x640) IN SEQUENCE")
        
        # Clear session frames to free memory
        session["frames"] = []
        session["hand_crops"] = []
        gc.collect()
        
        return {
            "ok": True,
            "total_frames": total_frames,
            "hand_frames": len(cropped_b64),
            "cropped_images": cropped_b64,
            "message": message
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error stopping capture: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Failed to stop capture: {str(e)}")
