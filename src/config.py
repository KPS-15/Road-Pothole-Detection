"""
Configuration module for Road Pothole Detection project.
Centralizes all file paths, model hyperparameters, data splitting ratios,
and dashcam inference settings. No hardcoded paths elsewhere in the codebase.
"""

from pathlib import Path
import os
import random
import numpy as np

# Base directory: resolved relative to this configuration file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directories
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Models and Artifact directories
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# Results and logs
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_PR_CURVES_PATH = RESULTS_DIR / "roc_pr_curves.png"
METRICS_SUMMARY_PATH = RESULTS_DIR / "metrics_summary.json"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"
TRAINING_LOG_CSV = RESULTS_DIR / "training_log.csv"
DATASET_DISTRIBUTION_CHART = RESULTS_DIR / "dataset_class_distribution.png"

# Outputs and media
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLE_VIDEOS_DIR = OUTPUTS_DIR / "sample_videos"
ANNOTATED_OUTPUT_VIDEO = OUTPUTS_DIR / "annotated_dashcam_output.mp4"

# Class definitions
# Note: Binary classification with Sigmoid unit outputting probability of class index 1 (Pothole)
CLASS_NAMES = ["Normal", "Pothole"]
CLASS_LABELS = {
    "Normal": 0,
    "Pothole": 1
}
POSITIVE_CLASS = "Pothole"
NEGATIVE_CLASS = "Normal"

# Model Architecture & Image Preprocessing Parameters
IMG_HEIGHT = 299
IMG_WIDTH = 299
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Data Splitting Parameters
TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# Training Hyperparameters
RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 16  # Tuned for laptop CPU/GPU memory safety
HEAD_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 1e-5
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
DROPOUT_RATE = 0.4
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-6

# Video Inference & Dashcam Parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_FRAME_SAMPLE_INTERVAL = 3  # Process every 3rd frame for real-time responsiveness

# Supported Image and Video Formats
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Kaggle Dataset Reference
KAGGLE_DATASET_ID = "atulyakumar98/pothole-detection-dataset"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset"


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Ensure reproducibility across python, numpy, and tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_directories() -> None:
    """Create all essential project directories if they do not already exist."""
    directories = [
        DATASET_DIR,
        TRAIN_DIR / "Normal",
        TRAIN_DIR / "Pothole",
        VAL_DIR / "Normal",
        VAL_DIR / "Pothole",
        TEST_DIR / "Normal",
        TEST_DIR / "Pothole",
        MODELS_DIR,
        RESULTS_DIR,
        OUTPUTS_DIR,
        SAMPLE_VIDEOS_DIR,
        RAW_DATA_DIR,
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
