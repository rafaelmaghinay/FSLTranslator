# FSL Translator Backend

A professional, production-ready FastAPI backend for Filipino Sign Language (FSL) recognition using deep learning. Provides REST API endpoints for image/video upload, real-time webcam detection, and gesture classification.

## 📋 Project Structure

```
backend-fastapi/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── core/                     # Configuration and constants
│   │   ├── __init__.py
│   │   ├── config.py             # Environment config, paths, API settings
│   │   └── constants.py          # FSL class names and gesture mappings
│   ├── routes/                   # API endpoint handlers
│   │   ├── __init__.py
│   │   ├── health.py             # Health check endpoints
│   │   ├── upload.py             # Image/video upload endpoints
│   │   ├── classify.py           # Gesture classification endpoint
│   │   ├── live.py               # Live webcam detection endpoints
│   │   └── management.py         # Admin/utility endpoints
│   ├── services/                 # Business logic and ML models
│   │   ├── __init__.py
│   │   ├── ml_model.py           # Model loading and inference
│   │   ├── classify_service.py   # Classification logic
│   │   └── webcam_service.py     # Webcam session management
│   ├── schemas/                  # Pydantic request/response models
│   │   ├── __init__.py
│   │   ├── requests.py           # Input validation models
│   │   └── responses.py          # Response models
│   └── utils/                    # Helper utilities
│       ├── __init__.py
│       ├── file_utils.py         # File I/O operations
│       └── hand_detection.py     # Hand detection utilities
├── models/                       # Pre-trained ML models
│   ├── bilstm_best_test14.pth   # BiLSTM classifier model
│   └── yolo_best.pt             # YOLO hand detector
├── uploads/                      # Uploaded files (auto-created)
├── main.py                       # FastAPI application entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container configuration
├── .dockerignore                # Docker ignore rules
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support, optional but recommended)
- 8GB+ RAM

### Installation

1. **Clone and navigate to backend:**
   ```bash
   cd backend-fastapi
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run development server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 🐳 Docker Deployment

```bash
# Build image
docker build -t fsl-translator-backend .

# Run container
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS="http://localhost:5173" \
  fsl-translator-backend
```

## 📡 API Endpoints

### Health Check
- `GET /` - Server status

### Image/Video Upload
- `POST /api/upload/sequence` - Process image sequence
- `POST /api/upload/video` - Process video file

### Classification
- `POST /api/classify` - Classify gesture from cropped images

### Live Detection
- `WebSocket /api/live/ws` - Real-time hand detection
- `POST /api/live/capture` - Start capture session
- `POST /api/live/frame` - Store frame for session
- `POST /api/live/stop` - Stop capture and get crops

### Management
- `POST /api/clear` - Clear all uploads

## 🧠 ML Models

### Hand Detector (YOLO)
- **Model:** YOLOv8 fine-tuned for hand detection
- **Input:** Video frames (any resolution)
- **Output:** Bounding boxes with confidence scores
- **Performance:** ~50 FPS on GPU, ~10 FPS on CPU

### Gesture Classifier (ResNet34 + BiLSTM)
- **Architecture:** ResNet34 feature extraction + Bidirectional LSTM
- **Input:** Sequence of 20 cropped hand images (224×224)
- **Output:** Predicted gesture with confidence score
- **Classes:** 43 gestures (alphabets, digits, common phrases)

## ⚙️ Configuration

Edit `app/core/config.py` to customize:
- `ALLOWED_ORIGINS` - CORS settings
- `BILSTM_MODEL_PATH`, `YOLO_MODEL_PATH` - Model paths
- `SEQ_LEN` - Sequence length for LSTM
- `YOLO_CONFIDENCE_THRESHOLD` - Detection threshold

## 🔒 Environment Variables

Optional environment variables:
```bash
ALLOWED_ORIGINS="http://localhost:5173,https://example.com"
```

## 📦 Dependencies

Core dependencies:
- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **torch** - Deep learning framework
- **ultralytics** - YOLO implementation
- **opencv-python** - Computer vision
- **pydantic** - Data validation
- **python-multipart** - Form data handling

See `requirements.txt` for complete list.

## 🧪 Performance Metrics

- **Image sequence processing:** 50-200ms (8+ images)
- **Video frame extraction:** 200-500ms (up to 30s videos)
- **Gesture classification:** 30-100ms (20-frame sequence)
- **Live hand detection:** 50-100ms per frame

## 🎯 Supported Gestures

### Alphabets (A-Z)
Single hand finger-spelled letters

### Digits (0-9)
Numerical sign language

### Common Phrases
- Good Morning
- Good Afternoon
- Good Evening
- How Are You?
- I'm Fine
- Thank You
- Sorry
- You're Welcome

## 📝 Development Guidelines

### Code Organization
- **Models** - Pure data classes (Pydantic)
- **Services** - Business logic (model inference, processing)
- **Routes** - HTTP handlers (requests/responses)
- **Utils** - Reusable helper functions

### Type Hints
All functions include type hints for clarity and IDE support.

### Error Handling
Proper HTTP status codes and descriptive error messages for all endpoints.

### Logging
Console logging with status emojis for debugging.

## 🚨 Troubleshooting

**CUDA errors:**
```bash
# Run on CPU only
export CUDA_VISIBLE_DEVICES=-1
```

**Memory issues:**
```bash
# Reduce batch size in config.py
# Clear uploads regularly via /api/clear endpoint
```

**Port already in use:**
```bash
uvicorn main:app --port 8001
```

## 📄 License

[Your License Here]

## 👤 Author

[Your Name]

---

**Made for recruiters** ✨
