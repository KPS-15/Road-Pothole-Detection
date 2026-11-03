"""
Main FastAPI application entry point.
Configures CORS, mounts static artifact directories, and registers modular API routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import (
    CORS_ORIGINS,
    RESULTS_DIR,
    OUTPUTS_DIR,
)
from backend.app.routes import health, predictions, videos
from backend.app.model_service import model_service

app = FastAPI(
    title="Road Pothole Detection API",
    description="Deep Learning REST API for road hazard and pothole classification from dashcam footage using InceptionV3.",
    version="1.0.0",
)

# CORS configuration for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static artifacts (for training curves, confusion matrix, and output videos)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/static/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Register API Routers
app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(videos.router)


@app.on_event("startup")
async def startup_event():
    print("[FastAPI] Service starting up...")
    available = model_service.is_available()
    if available:
        print("[FastAPI] InceptionV3 trained checkpoint detected. Ready for inference.")
    else:
        print("[FastAPI] Warning: Trained model checkpoint not found. Model routes will report unavailable until trained.")


@app.get("/")
def root():
    return {
        "service": "Road Pothole Detection REST API",
        "model": "InceptionV3 Deep CNN",
        "documentation": "/docs",
        "health": "/api/health",
        "model_status": "/api/model/status",
    }
