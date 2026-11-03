"""
Backend configuration settings for FastAPI service.
Manages file paths, CORS configuration, and inference defaults.
"""

from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
PROCESSED_VIDEOS_DIR = OUTPUTS_DIR / "processed"
SAMPLE_VIDEOS_DIR = OUTPUTS_DIR / "sample_videos"

# Ensure runtime directories exist
for p in [UPLOADS_DIR, PROCESSED_VIDEOS_DIR, SAMPLE_VIDEOS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Image & Model Specifications
IMG_HEIGHT = 299
IMG_WIDTH = 299
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
DEFAULT_THRESHOLD = 0.50
DEFAULT_FRAME_INTERVAL = 3

# Security & CORS
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_BYTES = 150 * 1024 * 1024  # 150MB limit
