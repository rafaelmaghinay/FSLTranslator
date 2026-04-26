"""
FSL Translator Backend - Main Application Entry Point

Filipino Sign Language Recognition API with FastAPI.
Provides endpoints for image/video upload, live webcam detection, and gesture classification.

Quick Start:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import API_TITLE, API_VERSION, API_DESCRIPTION, ALLOWED_ORIGINS, UPLOADS_DIR
from app.routes import health, upload, classify, live, management

# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for uploads
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Include routers
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(classify.router)
app.include_router(live.router)
app.include_router(management.router)

# Print startup information
if __name__ == "__main__":
    print(f"""
    🚀 {API_TITLE} v{API_VERSION}
    
    Starting FastAPI server...
    📖 API Documentation: http://localhost:8000/docs
    📚 ReDoc: http://localhost:8000/redoc
    
    Allowed origins:
    {chr(10).join(f'  • {origin}' for origin in ALLOWED_ORIGINS)}
    """)

