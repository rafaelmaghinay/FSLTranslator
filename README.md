# FSL Translator - Filipino Sign Language Recognition System

A full-stack web application for recognizing and translating Filipino Sign Language (FSL) gestures using deep learning and computer vision. The system processes hand movements through video/image sequences and classifies them using a ResNet34-BiLSTM neural network.

## 🎯 Project Overview

FSL Translator combines modern machine learning with an intuitive web interface to:
- **Detect hand regions** in images/videos using YOLO object detection
- **Extract frame sequences** with temporal sampling (10/80/10 strategy)
- **Classify gestures** using a ResNet34 backbone with bidirectional LSTM
- **Provide real-time recognition** through live webcam streaming
- **Support multiple input formats**: single images, 20-image sequences, videos, or live camera

## 🏗️ Architecture

### Technology Stack

**Backend:**
- FastAPI (Python web framework)
- PyTorch (Deep learning)
- YOLO v8 (Hand detection)
- OpenCV (Video/image processing)
- Pydantic (Data validation)

**Frontend:**
- React 19 (UI framework)
- Vite (Build tool)
- React Hooks (State management)
- Tailwind CSS (Styling)
- WebSocket (Real-time communication)

**ML Models:**
- ResNet34 (Feature extraction backbone)
- Bidirectional LSTM (Temporal sequence modeling)
- YOLO (Real-time hand detection)

### Project Structure

```
FSLTranslator/
├── backend-fastapi/                # Backend API server
│   ├── main.py                    # FastAPI application & endpoints
│   ├── upload.py                  # Image/video processing & classification
│   ├── webcam.py                  # Live camera session management
│   ├── models/                    # Pre-trained model weights
│   │   ├── bilstm_best_test14.pth # Classification model
│   │   └── yolo_best.pt           # Hand detection model
│   ├── uploads/                   # Temporary file storage
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Container configuration
│
└── fsl-translator/                 # Frontend application
    ├── src/
    │   ├── App.jsx                # Main routing component
    │   ├── config.js              # Server configuration
    │   ├── main.jsx               # React entry point
    │   ├── components/
    │   │   ├── Upload.jsx         # File upload interface
    │   │   ├── StartAnalysis.jsx  # Analysis preview
    │   │   ├── Results.jsx        # Classification results display
    │   │   ├── LiveCamera.jsx     # Real-time camera stream
    │   │   ├── NavigationBar.jsx  # Top navigation
    │   │   ├── ImageSlider.jsx    # Preview carousel
    │   │   ├── BackButton.jsx     # Navigation button
    │   │   └── About.jsx          # Project information
    │   └── assets/                # Images and static files
    ├── package.json               # NPM dependencies
    ├── vite.config.js             # Vite configuration
    └── index.html                 # HTML template
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (Backend)
- Node.js 16+ (Frontend)
- CUDA-capable GPU (optional, for faster inference)

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd backend-fastapi
   pip install -r requirements.txt
   ```

2. **Verify model files exist:**
   ```
   models/bilstm_best_test14.pth  (Classification model)
   models/yolo_best.pt             (Hand detection model)
   ```

3. **Run the server:**
   ```bash
   python main.py
   # or with Uvicorn directly
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at: `http://localhost:8000`

### Frontend Setup

1. **Install Node dependencies:**
   ```bash
   cd fsl-translator
   npm install
   ```

2. **Update server configuration (if needed):**
   Edit `src/config.js`:
   ```javascript
   export const SERVER_BASE = 'http://localhost:8000';
   export const WS_URL = 'ws://localhost:8000/api/live';
   ```

3. **Start development server:**
   ```bash
   npm run dev
   # Opens at http://localhost:5173
   ```

## 📊 API Endpoints

### Upload Endpoints

**POST `/api/upload/image`**
- Upload single image for hand detection and cropping
- Returns: Cropped hand region as base64

**POST `/api/upload/sequence`**
- Upload exactly 20 images for sequence analysis
- Returns: Array of cropped hand regions as base64

**POST `/api/upload/video`**
- Upload video file for frame extraction and hand detection
- Returns: Sampled frames with detected hands as base64

### Classification

**POST `/api/classify`**
- Classify gesture from base64-encoded cropped images
- Input: `{ "cropped_images": [base64_string, ...] }`
- Returns: Top predictions with confidence scores

### Utilities

**POST `/api/clear`**
- Delete all uploaded files from server

**GET `/`**
- Health check endpoint

### WebSocket

**WS `/api/live`**
- Real-time hand detection stream
- Send: Base64-encoded video frames
- Receive: Detected hand bounding boxes

## 🤖 Machine Learning Models

### ResNet34-BiLSTM Architecture

```
Input (20 frames × 3 × 224 × 224)
         ↓
  ResNet34 Backbone (frozen weights)
  Extracts features per frame
         ↓
  LSTM Layer (bidirectional)
  Captures temporal dependencies
         ↓
  Classification Head
  Dense layers with dropout
         ↓
  Output (43 sign classes)
```

**Key Features:**
- Pre-trained ResNet34 backbone preserves learned image features
- Bidirectional LSTM captures temporal context in both directions
- Dropout layers prevent overfitting on gesture sequences
- Supports 43 different signs: 26 alphabets, 10 digits, 7 common phrases

### Hand Detection (YOLO)

- Real-time hand localization in video frames
- Configurable confidence thresholds for accuracy/speed tradeoff
- Automatic frame resizing for optimal inference speed

## 📸 Processing Pipeline

### Image Sequence Processing
1. User uploads 20 images
2. YOLO detects hand regions in each frame
3. Extract cropped hands with consistent padding
4. Validate minimum hand detections (≥8)
5. Send to classification model

### Video Processing
1. Extract frames from video file
2. Sample frames uniformly to find hands
3. Apply 10/80/10 temporal sampling:
   - 10% from beginning frames
   - 80% from middle frames  
   - 10% from ending frames
4. Pad to exactly 20 frames if insufficient
5. Classify gesture

### Live Camera Stream
1. Establish WebSocket connection
2. Stream frames at real-time intervals
3. Receive hand detection boxes
4. Optional: Capture frames for batch analysis
5. Process captured sequence for classification

## 🎓 Classification Classes

The model recognizes:
- **Alphabets (26)**: A-Z in FSL
- **Digits (10)**: 0-9
- **Common Phrases (7)**:
  - Good Morning
  - Good Afternoon
  - Good Evening
  - How Are You
  - Thank You
  - You're Welcome
  - Sorry
  - I'm Fine

## 📈 Performance & Optimization

**Backend Optimizations:**
- Frame skipping in WebSocket (3-frame intervals)
- Batch YOLO processing for efficiency
- GPU acceleration with CUDA when available
- FP16 inference for reduced memory footprint

**Frontend Optimizations:**
- Lazy component loading
- Image compression for base64 transmission
- Websocket frame throttling
- Responsive UI with Tailwind CSS

**Deployment:**
- Docker containerization for consistent deployment
- CORS configuration for cross-origin requests
- Static file serving for media uploads

## 🧪 Development Features

### Logging & Debugging
- Detailed console logging for all processing steps
- Error tracking throughout pipeline
- Server response validation on frontend

### File Management
- Automatic cleanup of temporary files
- Organized upload directories by type (images, sequences, videos, temp)
- Safe disk space management

### Configuration
- Backend CORS settings for multiple domains
- Configurable YOLO confidence thresholds
- Model path management for easy updates

## 🔒 Error Handling

The system provides clear error messages for:
- Invalid file formats
- Insufficient hand detections
- Processing timeouts
- Network failures
- Model loading issues

## 💡 Key Implementation Highlights

1. **Adaptive Frame Sampling**: 10/80/10 strategy balances temporal coverage with focus on gesture peak
2. **Bidirectional LSTM**: Captures gesture context from both directions for more accurate classification
3. **YOLO Integration**: Fast hand detection enables real-time processing capability
4. **WebSocket Streaming**: Efficient real-time video stream communication
5. **Pydantic Validation**: Type-safe API request/response handling
6. **Graceful Degradation**: CPU fallback when GPU unavailable

## 📝 Usage Examples

### Uploading Images
1. Click "20 Image Sequence" card
2. Select exactly 20 images of FSL gesture
3. Click "Start Classification"
4. View results with confidence scores

### Using Live Camera
1. Click "Start Live Camera" card
2. Allow camera access
3. Perform FSL gesture in frame
4. Press space or click "Capture" to record
5. Click "Stop & Analyze" to classify

### Uploading Video
1. Click "Upload a Video" card
2. Select video file with FSL gesture
3. System automatically extracts and processes frames
4. View classification results

## 🛠️ Advanced Configuration

### Changing Model Paths
Edit in `upload.py`:
```python
LSTM_CHECKPOINT_PATH = "path/to/your/model.pth"
YOLO_MODEL_PATH = "path/to/your/yolo.pt"
```

### Adjusting Detection Confidence
In `webcam.py`:
```python
CONFIDENCE_THRESHOLD = 0.5  # Increase for stricter detection
```

### Video Frame Sampling
In `upload.py`:
```python
TARGET_VIDEO_FRAMES = 20  # Exact frames to extract
```

## 📚 Dependencies

See [requirements.txt](backend-fastapi/requirements.txt) for complete backend dependencies.

Frontend uses: React, Vite, Tailwind CSS (see [package.json](fsl-translator/package.json))

## 🤝 Contributing

Code follows best practices including:
- Professional function/class documentation
- Consistent error handling
- Clear variable naming conventions
- Separated concerns (processing, detection, classification)

## 📄 Project Information

**Purpose:** Educational & Production-Ready Sign Language Recognition

**Status:** Active Development

**Author:** Created as a comprehensive full-stack application demonstrating:
- Deep learning model deployment
- Real-time computer vision processing  
- WebSocket real-time communication
- Full production pipeline from data to UI

---

**For questions or issues, please review error logs and ensure:**
- Python/Node dependencies are installed
- Model files are present and accessible
- Backend/Frontend servers are running
- CORS is properly configured for your domain
