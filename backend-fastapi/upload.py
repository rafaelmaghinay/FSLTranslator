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

CLASS_NAMES = [
    "alphabets_A",  # Index 0
    "alphabets_B",  # Index 1
    "alphabets_C",  # Index 2
    "alphabets_D",  # Index 3
    "alphabets_E",  # Index 4
    "alphabets_F",  # Index 5
    "alphabets_G",  # Index 6
    "alphabets_H",  # Index 7
    "alphabets_I",  # Index 8
    "alphabets_J",  # Index 9
    "alphabets_K",  # Index 10
    "alphabets_L",  # Index 11
    "alphabets_M",  # Index 12
    "alphabets_N",  # Index 13
    "alphabets_O",  # Index 14
    "alphabets_P",  # Index 15
    "alphabets_Q",  # Index 16
    "alphabets_R",  # Index 17
    "alphabets_S",  # Index 18
    "alphabets_T",  # Index 19
    "alphabets_U",  # Index 20
    "alphabets_V",  # Index 21
    "alphabets_W",  # Index 22
    "alphabets_X",  # Index 23
    "alphabets_Y",  # Index 24
    "alphabets_Z",  # Index 25
    "digits_eight",  # Index 26
    "digits_five",  # Index 27
    "digits_four",  # Index 28
    "digits_nine",  # Index 29
    "digits_one",  # Index 30
    "digits_seven",  # Index 31
    "digits_six",  # Index 32
    "digits_three",  # Index 33
    "digits_two",  # Index 34
    "phrases_GoodAfternoon",  # Index 35
    "phrases_GoodEvening",  # Index 36
    "phrases_GoodMorning",  # Index 37
    "phrases_HowAreYou",  # Index 38
    "phrases_Imfine",  # Index 39
    "phrases_Sorry",  # Index 40
    "phrases_ThankYou",  # Index 41
    "phrases_YoureWelcome",  # Index 42
]

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
    checkpoint = torch.load("models/bilstm_best_test13.pth", map_location='cpu', weights_only=False)
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

        sample_count = min(total_frames, max(target_frames * 3, target_frames))
        sample_indices = np.linspace(0, total_frames - 1, sample_count).astype(int)

        candidates = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            results = detector(frame, verbose=False)
            if not results or len(results) == 0:
                continue

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            h, w = frame.shape[:2]
            valid_boxes = [box for box in r.boxes if float(box.conf[0]) >= 0.30]
            valid_boxes = sorted(valid_boxes, key=lambda b: float(b.xyxy[0][0]))
            
            for box in valid_boxes[:2]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
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

        # ⬅️ NEW: Handle no hands detected
        if len(candidates) == 0:
            return {
                "status": "error",
                "error": "No hands detected in any frame",
                "total_frames": total_frames
            }

        # ⬅️ NEW: Apply 10/80/10 padding if insufficient crops
        selected_crops = []
        
        if len(candidates) < target_frames:
            print(f"⚠️  Only {len(candidates)} crops found, padding to {target_frames} using 10/80/10 strategy...")
            
            # Sort by temporal order
            candidates.sort(key=lambda t: t[0])
            
            # Calculate 10/80/10 split
            total_crops = len(candidates)
            begin_end_idx = max(1, int(total_crops * 0.10))  # First 10%
            middle_start_idx = begin_end_idx
            middle_end_idx = total_crops - begin_end_idx  # Last 10%
            
            # Extract regions
            begin_crops = candidates[:begin_end_idx]
            middle_crops = candidates[middle_start_idx:middle_end_idx]
            end_crops = candidates[middle_end_idx:]
            
            print(f"   📦 Regions: {len(begin_crops)} begin, {len(middle_crops)} middle, {len(end_crops)} end")
            
            # Calculate how many we need from each region
            num_begin = int(target_frames * 0.10)  # 2 frames
            num_end = int(target_frames * 0.10)    # 2 frames
            num_middle = target_frames - num_begin - num_end  # 16 frames
            
            # Sample from each region (with padding if needed)
            def pad_region(region_crops, target_count):
                """Sample or pad a region to reach target_count."""
                if len(region_crops) == 0:
                    return []
                if len(region_crops) >= target_count:
                    # Uniform sampling
                    indices = np.linspace(0, len(region_crops) - 1, target_count, dtype=int)
                    return [region_crops[i] for i in indices]
                else:
                    # Pad by duplicating middle crops
                    result = region_crops.copy()
                    middle_idx = len(region_crops) // 2
                    while len(result) < target_count:
                        result.insert(middle_idx, region_crops[middle_idx])
                    return result
            
            # Sample each region
            sampled_begin = pad_region(begin_crops, num_begin)
            sampled_middle = pad_region(middle_crops, num_middle)
            sampled_end = pad_region(end_crops, num_end)
            
            # Combine in order
            selected_crops = [crop for _, crop in (sampled_begin + sampled_middle + sampled_end)]
            
            print(f"   ✅ Padded using 10/80/10: {len(sampled_begin)} begin + {len(sampled_middle)} middle + {len(sampled_end)} end")
        
        else:
            # ⬅️ ENOUGH CROPS: Apply 10/80/10 sampling
            print(f"✅ Found {len(candidates)} crops, sampling {target_frames} using 10/80/10 strategy...")
            
            candidates.sort(key=lambda t: t[0])
            total_crops = len(candidates)
            
            # Calculate split indices
            begin_end_idx = int(total_crops * 0.10)
            middle_end_idx = int(total_crops * 0.90)
            
            # Split into regions
            begin_crops = candidates[:begin_end_idx]
            middle_crops = candidates[begin_end_idx:middle_end_idx]
            end_crops = candidates[middle_end_idx:]
            
            # Calculate target counts
            num_begin = int(target_frames * 0.10)  # 2
            num_middle = int(target_frames * 0.80)  # 16
            num_end = target_frames - num_begin - num_middle  # 2
            
            print(f"   📊 Sampling: {num_begin} from begin, {num_middle} from middle, {num_end} from end")
            
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
            
            # Handle edge cases: if region too small, borrow from middle
            if len(sampled_begin) < num_begin:
                deficit = num_begin - len(sampled_begin)
                sampled_middle = sample_region(middle_crops, num_middle + deficit)
                print(f"   ⚠️ Begin region too small, borrowed {deficit} from middle")
            
            if len(sampled_end) < num_end:
                deficit = num_end - len(sampled_end)
                sampled_middle = sample_region(middle_crops, num_middle + deficit)
                print(f"   ⚠️ End region too small, borrowed {deficit} from middle")
            
            # Combine crops in temporal order
            selected_crops = [crop for _, crop in (sampled_begin + sampled_middle + sampled_end)]

        # Save crops
        req_dir = TEMP_DIR / f"video_{uuid.uuid4().hex}"
        req_dir.mkdir(parents=True, exist_ok=True)

        crop_paths = []
        for i, crop in enumerate(selected_crops):
            out_path = req_dir / f"frame_{i:03d}_crop.jpg"
            cv2.imwrite(str(out_path), crop)
            crop_paths.append(str(out_path))

        print(f"[3/3] Extracted {len(crop_paths)} cropped hand frames using 10/80/10 strategy")
        return {
            "status": "success",
            "cropped_images": crop_paths,
            "count": len(crop_paths),
            "total_frames": total_frames,
            "sampling_strategy": "10/80/10"  # ⬅️ NEW
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