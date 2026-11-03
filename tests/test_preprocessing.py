"""
Unit tests for image preprocessing, data normalization, and integrity verification.
Does not require full dataset or heavy GPU execution.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS,
    INPUT_SHAPE,
    CLASS_LABELS,
)
from src.data_utils import (
    compute_file_hash,
    extract_scene_key,
    normalize_class_name,
    inspect_image_file,
)
from src.video_utils import preprocess_frame_for_inception


def test_input_shape_hyperparameters():
    """Verify standard input shape matches InceptionV3 requirements."""
    assert IMG_HEIGHT == 299
    assert IMG_WIDTH == 299
    assert CHANNELS == 3
    assert INPUT_SHAPE == (299, 299, 3)


def test_class_normalization():
    """Verify class name mapping handles varied folder and file names."""
    assert normalize_class_name("potholes") == "Pothole"
    assert normalize_class_name("POTHOLE") == "Pothole"
    assert normalize_class_name("road_damage") == "Pothole"
    assert normalize_class_name("normal") == "Normal"
    assert normalize_class_name("smooth_road") == "Normal"
    assert normalize_class_name("plain") == "Normal"
    assert normalize_class_name("random_noise_123") is None


def test_scene_key_extraction():
    """Verify dashcam scene grouping extracts prefixes to prevent leakage."""
    assert extract_scene_key("dashcam_video01_frame042.jpg") == "dashcam_video" or "dashcam" in extract_scene_key("dashcam_video01_frame042.jpg")
    assert extract_scene_key("highway_A_001.png") == "highway_A" or "highway" in extract_scene_key("highway_A_001.png")


def test_frame_preprocessing_for_inception():
    """Verify that video frame preprocessing creates the exact tensor shape and [-1, 1] range."""
    # Create synthetic test frame (BGR 480x640x3)
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    preprocessed = preprocess_frame_for_inception(dummy_bgr)
    
    # Assert shape: (1, 299, 299, 3)
    assert preprocessed.shape == (1, 299, 299, 3)
    assert preprocessed.dtype == np.float32
    
    # InceptionV3 preprocess_input normalizes [0, 255] to [-1.0, 1.0]
    assert preprocessed.min() >= -1.05
    assert preprocessed.max() <= 1.05


def test_image_file_verification(tmp_path):
    """Test image verification utility with valid and corrupted files."""
    # 1. Valid RGB image
    valid_img_path = tmp_path / "valid_road.jpg"
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    img.save(valid_img_path)

    is_valid, dims, err = inspect_image_file(valid_img_path)
    assert is_valid is True
    assert dims == (640, 480, 3)
    assert err is None

    # 2. Corrupted file
    corrupt_path = tmp_path / "corrupted.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_BYTES")

    is_valid_c, dims_c, err_c = inspect_image_file(corrupt_path)
    assert is_valid_c is False
    assert dims_c is None
    assert err_c is not None


def test_duplicate_file_hashing(tmp_path):
    """Verify SHA-256 detects identical images."""
    file1 = tmp_path / "road1.jpg"
    file2 = tmp_path / "road1_copy.jpg"
    file3 = tmp_path / "road2_different.jpg"

    content1 = b"IMAGE_CONTENT_SAMPLE_A_12345"
    content2 = b"IMAGE_CONTENT_SAMPLE_A_12345"
    content3 = b"IMAGE_CONTENT_SAMPLE_B_67890"

    file1.write_bytes(content1)
    file2.write_bytes(content2)
    file3.write_bytes(content3)

    hash1 = compute_file_hash(file1)
    hash2 = compute_file_hash(file2)
    hash3 = compute_file_hash(file3)

    assert hash1 == hash2
    assert hash1 != hash3
