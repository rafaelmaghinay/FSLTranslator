# Architecture Overview

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React/Vite)                       │
└────────────┬──────────────────────────┬─────────────────────────┘
             │                          │
             │ HTTP + WebSocket         │
             ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
│                     (app/routes/*)                              │
├────────┬────────┬──────────┬──────────┬──────────────────────────┤
│ Health │ Upload │ Classify │   Live   │   Management            │
│ Check  │ Routes │  Route   │  Routes  │   (Admin)               │
└────────┴───┬────┴────┬─────┴────┬─────┴──────────────────────────┘
             │        │          │
             ▼        ▼          ▼
┌────────────────────────────────────────────────────────────────┐
│                    Services Layer                              │
│              (app/services/*, app/utils/*)                     │
├──────────────┬──────────────┬──────────────────────────────────┤
│  ML Models   │ Classification│  Session Management             │
│  & Loading   │    Logic      │  & File Handling               │
└──────────┬───┴────┬─────┬────┴──────────┬───────────────────────┘
           │        │     │              │
           ▼        ▼     ▼              ▼
      ┌─────────┐  ┌──────────────┐  ┌──────────────┐
      │  YOLO   │  │  ResNet34+   │  │   File I/O   │
      │ Detector│  │   BiLSTM     │  │   & Cleanup  │
      └─────────┘  │  Classifier  │  └──────────────┘
                   └──────────────┘
           │                │
           ▼                ▼
    ┌────────────┐    ┌──────────────┐
    │  Pre-train │    │ Pre-trained  │
    │ YOLO Model │    │ ResNet34     │
    └────────────┘    │ + BiLSTM     │
                      │ Weights      │
                      └──────────────┘
```

## Request Flow Diagram

### Image Upload Flow
```
User Upload (Images)
        │
        ▼
/api/upload/sequence
        │
        ├─→ Save files (app/utils/file_utils.py)
        │
        ├─→ Load images (cv2)
        │
        ├─→ Detect hands (app/utils/hand_detection.py)
        │       └─→ Use YOLO model (app/services/ml_model.py)
        │
        ├─→ Crop hand regions (batch processing)
        │
        └─→ Return base64-encoded crops
```

### Classification Flow
```
Cropped Hand Sequence
        │
        ▼
/api/classify
        │
        ├─→ Validate request (app/schemas/requests.py)
        │
        ├─→ Decode base64 images
        │
        ├─→ Preprocess images (app/services/classify_service.py)
        │       ├─→ Resize to 224×224
        │       └─→ Normalize with ImageNet stats
        │
        ├─→ Inference (app/services/ml_model.py)
        │       ├─→ ResNet34 feature extraction
        │       ├─→ BiLSTM temporal modeling
        │       └─→ Classification head
        │
        └─→ Return prediction + confidence (app/schemas/responses.py)
```

### Live Detection Flow
```
Browser Webcam Stream
        │
        ▼
WebSocket /api/live/ws
        │ (Frame visualization only)
        │
        ├─→ Receive base64 frame
        ├─→ Skip frames for performance
        ├─→ Detect hands (live.py:detect_boxes_yolo)
        └─→ Return bounding boxes

Browser Record Request
        │
        ▼
POST /api/live/capture
        │
        └─→ Create session (app/services/webcam_service.py)

Repeated:
        │
        ▼
POST /api/live/frame
        │
        ├─→ Decode frame
        ├─→ Store in session
        └─→ Return frame count

        │
        ▼
POST /api/live/stop
        │
        ├─→ Stop recording
        ├─→ Batch detect hands (all frames with YOLO)
        ├─→ Apply 10/80/10 temporal sampling
        ├─→ Crop hand regions
        └─→ Return base64 crops for classification
```

## Data Model Diagram

```
┌─────────────────────────────────────┐
│       ClassifyRequest                │
├─────────────────────────────────────┤
│ - cropped_images: List[str]         │  (base64-encoded)
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│    classify_service.py               │
│ - preprocess_image()                │
│ - classify_sequence()               │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│        ml_model.py                   │
│ - ResNet34_BiLSTM model             │
│ - ModelManager (singleton)          │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      ClassifyResponse                │
├─────────────────────────────────────┤
│ - ok: bool                          │
│ - prediction: str                   │
│ - confidence: float                 │
│ - all_predictions: List[dict]       │
└─────────────────────────────────────┘
```

## Configuration Hierarchy

```
app/core/config.py
    ├─ API_TITLE, API_VERSION
    ├─ ALLOWED_ORIGINS (CORS)
    ├─ MODEL_PATHS
    │   ├─ BILSTM_MODEL_PATH
    │   └─ YOLO_MODEL_PATH
    ├─ DIRECTORY_PATHS
    │   ├─ IMG_DIR
    │   ├─ SEQ_DIR
    │   ├─ VID_DIR
    │   └─ TEMP_DIR
    └─ ML_PARAMETERS
        ├─ SEQ_LEN = 20
        ├─ IMG_SIZE = 224
        ├─ LSTM_HIDDEN = 128
        └─ NUM_CLASSES = 43

app/core/constants.py
    ├─ CLASS_NAMES (list of 43 gestures)
    └─ DISPLAY_NAMES (mapping to readable names)
```

## Error Handling Flow

```
API Request
    │
    ▼
Route Handler (app/routes/*)
    │
    ├─→ Validate request (Pydantic)
    │       └─ If invalid → 422 Unprocessable Entity
    │
    ├─→ Call service
    │       │
    │       ├─→ If not found → 404 Not Found
    │       ├─→ If validation fails → 400 Bad Request
    │       └─→ If error → 500 Internal Server Error
    │
    └─→ Return response (app/schemas/responses.py)
        └─ JSON with proper status codes
```

## Performance Optimization Points

```
1. Live Detection (/api/live/ws)
   ├─ Frame skipping (every Nth frame)
   ├─ Resizing frames before YOLO
   └─ Caching last detection results

2. Video Processing
   ├─ Batch YOLO detection (multiple frames)
   ├─ 10/80/10 temporal sampling
   └─ Memory cleanup after processing

3. Model Loading
   ├─ Singleton pattern (loaded once)
   ├─ GPU acceleration (CUDA if available)
   └─ Model caching in memory

4. Image Processing
   ├─ Batch base64 encoding/decoding
   ├─ JPEG compression for storage
   └─ Memory-mapped file access
```

## Dependency Injection Pattern

```
main.py
    │
    ├─→ Imports routes from app/routes/
    ├─→ Imports services from app/services/
    ├─→ Imports config from app/core/
    │
    ▼
app/routes/*.py
    │
    ├─→ Import from app/services/
    ├─→ Import from app/schemas/
    └─→ Import from app/utils/

app/services/*.py
    │
    ├─→ Import from app/core/config.py
    └─→ Create/use service instances
```

This architecture ensures:
- ✅ Clean separation of concerns
- ✅ Testability (each layer can be tested independently)
- ✅ Reusability (services can be used by multiple routes)
- ✅ Maintainability (changes localized to relevant modules)
- ✅ Scalability (easy to add new routes/services)
