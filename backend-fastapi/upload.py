import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import traceback
import uuid
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR = UPLOADS_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION - MUST MATCH TRAINING SCRIPT
# ============================================================

SEQ_LEN = 20
IMG_SIZE = 224
LSTM_HIDDEN = 128
CLASSIFIER_DROPOUT = 0.4

CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "GoodAfternoon", "GoodEvening", "GoodMorning", "Hello", "HowAreYou", "Imfine", "Sorry", "ThankYou", "WhatsYourName", "YoureWelcome"]
NUM_CLASSES = len(CLASS_NAMES)

YOLO_BATCH_SIZE = 8
TARGET_VIDEO_FRAMES = 20

# ============================================================
# MODEL DEFINITION
# ============================================================

class ResNet34_BiLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden=96, freeze_backbone=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT
        resnet = models.resnet34(weights=weights)
        self.feat_dim = 512
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        feats = self.backbone(x_flat)
        feats = feats.view(B, S, -1)
        lstm_out, _ = self.lstm(feats)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)

# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
# MODEL LOADING
# ============================================================

print("🚀 Loading models...")

clf = ResNet34_BiLSTM(
    num_classes=NUM_CLASSES,
    lstm_hidden=LSTM_HIDDEN,
    freeze_backbone=True
)

try:
    checkpoint = torch.load("models/bilstm_best_test9.pth", map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        clf.load_state_dict(checkpoint['model_state_dict'])
    else:
        clf.load_state_dict(checkpoint)
    print("✅ ResNet34+BiLSTM model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

clf.eval()

detector = YOLO("models/yolo.pt")
print("✅ YOLO detector loaded")
print("✅ All models ready\n")

# ============================================================
# CORE PROCESSING FUNCTIONS
# ============================================================

def detect_and_crop(frame):
    """Detect hand in single frame and crop it."""
    results = detector(frame, stream=True, verbose=False)
    
    for r in results:
        if len(r.boxes) == 0:
            return None
        
        best = max(r.boxes, key=lambda b: b.conf[0])
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        
        pad = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        return frame[y1:y2, x1:x2]
    
    return None

def detect_and_crop_batch(frames):
    """Detect hands in batch of frames."""
    crops = []
    results = detector(frames, stream=False, verbose=False)
    
    for i, r in enumerate(results):
        if len(r.boxes) == 0:
            crops.append(None)
            continue
        
        best = max(r.boxes, key=lambda b: b.conf[0])
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        
        pad = 20
        h, w = frames[i].shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        crops.append(frames[i][y1:y2, x1:x2])
    
    return crops

def preprocess_crop(crop):
    """Convert cropped image to tensor and PIL Image."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return transform(pil_img), pil_img

def sample_frames_uniformly(frames, target_count):
    """Sample exactly target_count frames uniformly from list."""
    if len(frames) == 0:
        return []
    
    if len(frames) <= target_count:
        result = frames.copy()
        while len(result) < target_count:
            result.append(frames[-1])
        return result
    
    indices = np.linspace(0, len(frames) - 1, target_count).astype(int)
    return [frames[i] for i in indices]

def classify_sequence(frames):
    """Classify a sequence of frames."""
    seq = torch.stack(frames).unsqueeze(0)
    with torch.no_grad():
        out = clf(seq)
        probs = F.softmax(out, 1)
        conf, pred = torch.max(probs, 1)
        return CLASS_NAMES[pred.item()], float(conf.item())

# ============================================================
# HIGH-LEVEL PROCESSING FUNCTIONS
# ============================================================

def process_single_image(image_path: str) -> dict:
    """
    Process single image: detect hand and return cropped image info.
    
    Returns:
        dict: {
            "status": "success" | "no_hand_detected",
            "cropped_image_path": str (if success),
            "error": str (if error)
        }
    """
    try:
        frame = cv2.imread(str(image_path))
        if frame is None:
            return {"status": "error", "error": "Could not load image"}
        
        crop = detect_and_crop(frame)
        if crop is None:
            return {"status": "no_hand_detected"}
        
        # Save cropped image temporarily
        crop_path = str(image_path).replace(".jpg", "_crop.jpg").replace(".png", "_crop.png")
        cv2.imwrite(crop_path, crop)
        
        return {
            "status": "success",
            "cropped_image_path": crop_path,
            "original_image_path": str(image_path)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def process_image_sequence(image_paths: list) -> dict:
    """
    Process multiple images: detect hands and return cropped images info.
    
    Returns:
        dict: {
            "status": "success" | "insufficient_hands" | "error",
            "cropped_images": list of str (paths),
            "count": int,
            "error": str (if error)
        }
    """
    try:
        print(f"[1/2] Loading {len(image_paths)} images...")
        
        all_frames = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is not None:
                all_frames.append(img)
        
        if len(all_frames) == 0:
            return {"status": "error", "error": "Could not load any images"}
        
        print(f"[2/2] Detecting hands in {len(all_frames)} images...")
        
        all_crops = []
        crop_paths = []
        
        for i in range(0, len(all_frames), YOLO_BATCH_SIZE):
            batch = all_frames[i:i + YOLO_BATCH_SIZE]
            crops = detect_and_crop_batch(batch)
            
            for j, crop in enumerate(crops):
                if crop is not None and crop.size > 0:
                    img_idx = i + j
                    orig_path = image_paths[img_idx]
                    crop_path = str(orig_path).replace(".jpg", f"_crop_{j}.jpg").replace(".png", f"_crop_{j}.png")
                    cv2.imwrite(crop_path, crop)
                    all_crops.append(crop)
                    crop_paths.append(crop_path)
        
        if len(all_crops) < 8:
            return {
                "status": "insufficient_hands",
                "count": len(all_crops),
                "required": 8
            }
        
        return {
            "status": "success",
            "cropped_images": crop_paths,
            "count": len(all_crops),
            "original_count": len(all_frames)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def process_video(video_path: str, target_frames: int = TARGET_VIDEO_FRAMES) -> dict:
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"status": "error", "error": "Could not open video file"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        print(f"\n[1/3] VIDEO: {total_frames} frames, {fps:.1f} FPS")
        print("[2/3] Sampling frames uniformly to find hands...")

        if total_frames <= 0:
            cap.release()
            return {"status": "error", "error": "Video has 0 frames"}

        # oversample to increase chance of finding 20 hand frames
        sample_count = min(total_frames, max(target_frames * 3, target_frames))
        sample_indices = np.linspace(0, total_frames - 1, sample_count).astype(int)

        candidates = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # YOLO inference (single frame)
            results = detector(frame, verbose=False)
            if not results or len(results) == 0:
                continue

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # pick highest-confidence box
            best = max(r.boxes, key=lambda b: float(b.conf[0]))
            conf = float(best.conf[0])
            if conf < 0.30:
                continue

            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            h, w = frame.shape[:2]
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            candidates.append((int(idx), crop))

        cap.release()

        if len(candidates) < target_frames:
            return {
                "status": "insufficient_hands",
                "count": len(candidates),
                "required": target_frames,
                "total_frames": total_frames,
                "error": f"Only {len(candidates)} frames with hands detected",
            }

        # Keep temporal order; pick exactly target_frames uniformly among candidates
        candidates.sort(key=lambda t: t[0])
        pick_idx = np.linspace(0, len(candidates) - 1, target_frames).astype(int)
        selected = [candidates[i][1] for i in pick_idx]

        # Save crops
        req_dir = TEMP_DIR / f"video_{uuid.uuid4().hex}"
        req_dir.mkdir(parents=True, exist_ok=True)

        crop_paths = []
        for i, crop in enumerate(selected):
            out_path = req_dir / f"frame_{i:03d}_crop.jpg"
            cv2.imwrite(str(out_path), crop)
            crop_paths.append(str(out_path))

        print(f"[3/3] Extracted {len(crop_paths)} cropped hand frames")
        return {
            "status": "success",
            "cropped_images": crop_paths,
            "count": len(crop_paths),
            "total_frames": total_frames,
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
def classify_cropped_images(image_paths: list) -> dict:
    """
    Classify a sequence of cropped images.
    
    Returns:
        dict: {
            "status": "success" | "error",
            "prediction": str,
            "confidence": float,
            "top_3": list of dict with {label, confidence},  # ⬅️ NEW
            "frames_used": int,
            "error": str (if error)
        }
    """
    try:
        print(f"\n[1/2] Loading {len(image_paths)} images...")
        
        tensors = []
        for i, path in enumerate(image_paths):
            img = cv2.imread(str(path))
            if img is None:
                print(f"   ⚠ Could not load: {path}")
                continue
            
            tensor, _ = preprocess_crop(img)
            tensors.append(tensor)
        
        if len(tensors) == 0:
            return {"status": "error", "error": "Could not load any images"}
        
        # Handle sequence length
        if len(tensors) < SEQ_LEN:
            print(f"\n   Duplicating frames to reach {SEQ_LEN}...")
            selected_tensors = tensors.copy()
            while len(selected_tensors) < SEQ_LEN:
                for tensor in tensors:
                    if len(selected_tensors) >= SEQ_LEN:
                        break
                    selected_tensors.append(tensor)
        elif len(tensors) > SEQ_LEN:
            print(f"\n   Selecting {SEQ_LEN} frames from {len(tensors)}...")
            selected_tensors = sample_frames_uniformly(tensors, SEQ_LEN)
        else:
            selected_tensors = tensors
        
        print(f"[2/2] Classifying {len(selected_tensors)} frames...")
        
        # ⬅️ MODIFIED: Get top 3 predictions
        seq = torch.stack(selected_tensors).unsqueeze(0)
        with torch.no_grad():
            out = clf(seq)
            probs = F.softmax(out, 1)
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probs[0], 3)
            
            # Top 1 (best prediction)
            pred = CLASS_NAMES[top3_indices[0].item()]
            conf = float(top3_probs[0].item())
            
            # Top 3 results
            top_3 = [
                {
                    "label": CLASS_NAMES[top3_indices[i].item()],
                    "confidence": round(float(top3_probs[i].item()), 4)
                }
                for i in range(3)
            ]
        
        print(f"✅ Classification complete!")
        print(f"   Top 1: {pred} ({conf:.2%})")
        print(f"   Top 2: {top_3[1]['label']} ({top_3[1]['confidence']:.2%})")
        print(f"   Top 3: {top_3[2]['label']} ({top_3[2]['confidence']:.2%})")
        
        return {
            "status": "success",
            "prediction": pred,
            "confidence": conf,
            "top_3": top_3,  # ⬅️ NEW: Return top 3
            "frames_used": len(selected_tensors)
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}