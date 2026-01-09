from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import shutil
import time

app = FastAPI(title="FSL Upload Backend")

# Allow requests from Vite React dev server
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

for d in [IMG_DIR, SEQ_DIR, VID_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def save_file(upload: UploadFile, dest: Path):
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

@app.get("/")
def health():
    return {"ok": True, "message": "Backend is running"}

# 1) Single image
@app.post("/api/upload/image")
async def upload_image(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files allowed")

    ext = Path(image.filename).suffix
    filename = f"{int(time.time()*1000)}_{Path(image.filename).stem}{ext}"
    path = IMG_DIR / filename

    save_file(image, path)

    return {"ok": True, "type": "image", "saved_as": filename}

# 2) Image sequence (multiple)
@app.post("/api/upload/sequence")
async def upload_sequence(images: List[UploadFile] = File(...)):
    if not images:
        raise HTTPException(400, "No images uploaded")

    batch = SEQ_DIR / f"seq_{int(time.time()*1000)}"
    batch.mkdir(parents=True, exist_ok=True)

    saved = []
    for img in images:
        if not img.content_type or not img.content_type.startswith("image/"):
            raise HTTPException(400, "All files must be images")

        ext = Path(img.filename).suffix
        filename = f"{int(time.time()*1000)}_{Path(img.filename).stem}{ext}"
        path = batch / filename
        save_file(img, path)
        saved.append(filename)

    return {"ok": True, "type": "sequence", "count": len(saved), "folder": str(batch.name), "files": saved}

# 3) Video
@app.post("/api/upload/video")
async def upload_video(video: UploadFile = File(...)):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(400, "Only video files allowed")

    ext = Path(video.filename).suffix
    filename = f"{int(time.time()*1000)}_{Path(video.filename).stem}{ext}"
    path = VID_DIR / filename

    save_file(video, path)

    return {"ok": True, "type": "video", "saved_as": filename}
