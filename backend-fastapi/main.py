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
                
                # Frame skipping
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
                
                # ⬅️ USE resize_input=True for LIVE detection (faster)
                boxes = yolo_detect_boxes(frame, conf_thres=0.25, resize_input=True)
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
        session["timestamps"] = []  # ⬅️ NEW: Store timestamps
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
    frame: str = Form(...),
    timestamp: float = Form(None)  # ⬅️ NEW: Accept optional client timestamp
):
    """
    OPTIMIZED: Just store frames with timestamps, defer YOLO detection to /api/live/stop.
    NOW ACCEPTS CLIENT-SIDE TIMESTAMPS for more accurate timing!
    
    Args:
        session_id: Session identifier
        frame: Base64 encoded frame
        timestamp: Client-side timestamp (milliseconds since epoch) - OPTIONAL
    
    Returns:
        dict: {"ok": True, "total_frames": int, "timestamp": float}
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
        
        # ⬅️ USE CLIENT TIMESTAMP IF PROVIDED, OTHERWISE SERVER TIMESTAMP
        if timestamp is not None:
            # Client sent timestamp in milliseconds, convert to seconds
            client_timestamp = timestamp / 1000.0
            
            # Initialize session start time if not set
            if session.get("start_time") is None:
                session["start_time"] = client_timestamp
            
            relative_timestamp = client_timestamp - session["start_time"]
            used_timestamp = client_timestamp
            timestamp_source = "client"
        else:
            # Fallback to server timestamp
            server_timestamp = time.time()
            
            if session.get("start_time") is None:
                session["start_time"] = server_timestamp
            
            relative_timestamp = server_timestamp - session["start_time"]
            used_timestamp = server_timestamp
            timestamp_source = "server"
        
        # ⬅️ STORE FRAME AND TIMESTAMP
        session["frames"].append(frame_bgr.copy())
        session["timestamps"].append(used_timestamp)
        
        # Return immediately (much faster!)
        return {
            "ok": True,
            "total_frames": len(session["frames"]),
            "timestamp": used_timestamp,
            "relative_time": round(relative_timestamp, 3),
            "timestamp_source": timestamp_source  # ⬅️ NEW: Tell client which timestamp was used
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
    NOW USES 10/80/10 SAMPLING STRATEGY (beginning/middle/end).
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
        timestamps = session.get("timestamps", [])
        total_frames = len(frames)
        
        print(f"⏹️ Stopped capture session: {session_id}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Total frames captured: {total_frames}")
        print(f"   Frame size: {frames[0].shape if frames else 'N/A'}")
        print(f"   Timestamps collected: {len(timestamps)}")
        
        if total_frames == 0:
            return {
                "ok": False,
                "error": "No frames captured",
                "total_frames": 0,
                "hand_frames": 0
            }
        
        # ⬅️ PROCESS ALL FRAMES AT ORIGINAL SIZE - DETECT BOTH HANDS
        print(f"🔍 Processing {total_frames} frames with YOLO (at original resolution)...")
        
        all_hand_crops = []
        frame_indices = []
        crop_timestamps = []
        
        for idx, frame in enumerate(frames):
            boxes = yolo_detect_boxes(frame, conf_thres=0.30, resize_input=False)
            
            if len(boxes) == 0:
                continue
            
            h, w = frame.shape[:2]
            boxes_sorted = sorted(boxes, key=lambda b: b["x1"])
            
            frame_crops = []
            for box in boxes_sorted:
                x1 = int(box["x1"])
                y1 = int(box["y1"])
                x2 = int(box["x2"])
                y2 = int(box["y2"])
                
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    frame_crops.append(crop.copy())
            
            if len(frame_crops) > 0:
                all_hand_crops.extend(frame_crops)
                frame_indices.extend([idx] * len(frame_crops))
                crop_timestamps.extend([timestamps[idx] if idx < len(timestamps) else 0] * len(frame_crops))
        
        print(f"   ✅ Detected {len(all_hand_crops)} hand crops across {len(set(frame_indices))} frames")
        print(f"   Hands per frame: {len(all_hand_crops) / max(len(set(frame_indices)), 1):.1f} average")
        
        # ⬅️ NEW: 10/80/10 SAMPLING STRATEGY
        selected_crops = []
        selected_timestamps = []
        
        if len(all_hand_crops) == 0:
            return {
                "ok": False,
                "error": "No hands detected in any frame",
                "total_frames": total_frames,
                "hand_frames": 0
            }
        
        if len(all_hand_crops) < SEQ_LEN:
            # Not enough crops - pad with last crop
            original_count = len(all_hand_crops)
            selected_crops = all_hand_crops.copy()
            selected_timestamps = crop_timestamps.copy()
            
            while len(selected_crops) < SEQ_LEN:
                selected_crops.append(selected_crops[-1])
                selected_timestamps.append(selected_timestamps[-1])
            
            message = f"⚠️ Only {original_count} hand crops detected. Padded to {SEQ_LEN}."
            print(f"   {message}")
        
        else:
            # ⬅️ APPLY 10/80/10 SAMPLING STRATEGY
            total_crops = len(all_hand_crops)
            
            # Calculate split counts (10% / 80% / 10%)
            num_begin = int(SEQ_LEN * 0.10)  # 2 frames
            num_middle = int(SEQ_LEN * 0.80)  # 16 frames
            num_end = SEQ_LEN - num_begin - num_middle  # 2 frames (ensure exact SEQ_LEN)
            
            print(f"   📊 10/80/10 Sampling: {num_begin} beginning + {num_middle} middle + {num_end} end = {SEQ_LEN}")
            
            # Define temporal regions based on crop timestamps
            if crop_timestamps:
                min_time = min(crop_timestamps)
                max_time = max(crop_timestamps)
                time_range = max_time - min_time
                
                # Time boundaries: 10% | 80% | 10%
                begin_end_time = min_time + (time_range * 0.10)
                middle_end_time = min_time + (time_range * 0.90)
                
                print(f"   ⏱️ Time range: {time_range:.2f}s")
                print(f"   ⏱️ Begin: 0.00s - {begin_end_time - min_time:.2f}s")
                print(f"   ⏱️ Middle: {begin_end_time - min_time:.2f}s - {middle_end_time - min_time:.2f}s")
                print(f"   ⏱️ End: {middle_end_time - min_time:.2f}s - {time_range:.2f}s")
                
                # Split crops into temporal regions
                begin_crops = [(i, c, t) for i, (c, t) in enumerate(zip(all_hand_crops, crop_timestamps)) 
                              if t < begin_end_time]
                middle_crops = [(i, c, t) for i, (c, t) in enumerate(zip(all_hand_crops, crop_timestamps)) 
                               if begin_end_time <= t < middle_end_time]
                end_crops = [(i, c, t) for i, (c, t) in enumerate(zip(all_hand_crops, crop_timestamps)) 
                            if t >= middle_end_time]
                
                print(f"   📦 Region crops: {len(begin_crops)} begin, {len(middle_crops)} middle, {len(end_crops)} end")
                
                # Sample uniformly from each region
                def sample_region(region_crops, num_samples):
                    if len(region_crops) == 0:
                        return []
                    if len(region_crops) <= num_samples:
                        return region_crops
                    indices = np.linspace(0, len(region_crops) - 1, num_samples, dtype=int)
                    return [region_crops[i] for i in indices]
                
                sampled_begin = sample_region(begin_crops, num_begin)
                sampled_middle = sample_region(middle_crops, num_middle)
                sampled_end = sample_region(end_crops, num_end)
                
                # Handle edge cases: if a region has no crops, borrow from middle
                if len(sampled_begin) < num_begin:
                    deficit = num_begin - len(sampled_begin)
                    sampled_middle = sample_region(middle_crops, num_middle + deficit)
                    print(f"   ⚠️ Begin region has only {len(sampled_begin)} crops, borrowed {deficit} from middle")
                
                if len(sampled_end) < num_end:
                    deficit = num_end - len(sampled_end)
                    sampled_middle = sample_region(middle_crops, num_middle + deficit)
                    print(f"   ⚠️ End region has only {len(sampled_end)} crops, borrowed {deficit} from middle")
                
                # Combine in order: begin -> middle -> end
                sampled_all = sampled_begin + sampled_middle + sampled_end
                
                # Extract crops and timestamps (sorted by original index to preserve order)
                sampled_all_sorted = sorted(sampled_all, key=lambda x: x[0])
                selected_crops = [crop for _, crop, _ in sampled_all_sorted]
                selected_timestamps = [ts for _, _, ts in sampled_all_sorted]
                
                print(f"   ✅ Sampled {len(selected_crops)} crops using 10/80/10 strategy")
                print(f"   📊 Crop indices: {[idx for idx, _, _ in sampled_all_sorted]}")
                
                message = f"✅ Sampled {SEQ_LEN} crops using 10/80/10 strategy (begin/middle/end)"
            
            else:
                # Fallback: no timestamps, use uniform sampling
                print("   ⚠️ No timestamps available, falling back to uniform sampling")
                indices = np.linspace(0, len(all_hand_crops) - 1, SEQ_LEN, dtype=int)
                selected_crops = [all_hand_crops[i] for i in indices]
                selected_timestamps = [crop_timestamps[i] if i < len(crop_timestamps) else 0 for i in indices]
                message = f"✅ Sampled {SEQ_LEN} crops uniformly (no timestamps)"
        
        # Convert crops to base64
        cropped_b64 = []
        for i, crop in enumerate(selected_crops):
            _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buffer).decode()
            cropped_b64.append(f"data:image/jpeg;base64,{crop_b64}")
        
        # Calculate timing statistics
        if selected_timestamps:
            start_time = session["start_time"]
            relative_timestamps = [t - start_time for t in selected_timestamps]
            avg_fps = len(cropped_b64) / (relative_timestamps[-1] - relative_timestamps[0]) if len(relative_timestamps) > 1 else 0
        else:
            relative_timestamps = []
            avg_fps = 0
        
        print(f"   {message}")
        print(f"   Returning {len(cropped_b64)} cropped hand images IN SEQUENCE")
        print(f"   Average FPS of selected crops: {avg_fps:.1f}")
        
        # Clear session frames to free memory
        session["frames"] = []
        session["timestamps"] = []
        session["hand_crops"] = []
        gc.collect()
        
        return {
            "ok": True,
            "total_frames": total_frames,
            "hand_frames": len(cropped_b64),
            "cropped_images": cropped_b64,
            "timestamps": selected_timestamps,
            "relative_timestamps": relative_timestamps,
            "duration": duration,
            "avg_fps": round(avg_fps, 2),
            "sampling_strategy": "10/80/10",  # ⬅️ UPDATED: Changed to 10/80/10
            "message": message
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error stopping capture: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Failed to stop capture: {str(e)}")
