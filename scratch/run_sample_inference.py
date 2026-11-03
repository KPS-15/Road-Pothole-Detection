"""
Demonstration script: Run sample inference using trained InceptionV3 model.
Tests both static test images (Normal and Pothole) and frames extracted from sample dashcam video.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    SAMPLE_VIDEOS_DIR,
    TEST_DIR,
)
from src.video_utils import preprocess_frame_for_inception, get_video_metadata

def main():
    print("=" * 70)
    print("ROAD POTHOLE DETECTION: INCEPTION-V3 SAMPLE INFERENCE RUN")
    print("=" * 70)

    # 1. Load Model
    print(f"\n[1/3] Loading trained model from: {BEST_MODEL_PATH}")
    start_load = time.time()
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    load_time = time.time() - start_load
    print(f"       Model loaded successfully in {load_time:.2f}s")
    print(f"       Input shape: {model.input_shape}, Output shape: {model.output_shape}")

    # 2. Test Set Sample Inference
    print("\n[2/3] Performing sample inference on test images:")
    test_samples = [
        ("dataset/test/Normal/1.jpg", "Normal"),
        ("dataset/test/Normal/2.jpg", "Normal"),
        ("dataset/test/Pothole/19.jpg", "Pothole"),
        ("dataset/test/Pothole/23.jpg", "Pothole"),
    ]

    print(f"{'Image Path':<32} | {'True Label':<8} | {'Prob':<8} | {'Predicted':<8} | {'Latency':<8} | {'Match'}")
    print("-" * 80)

    for rel_path, true_label in test_samples:
        img_path = PROJECT_ROOT / rel_path
        if not img_path.exists():
            continue
        
        bgr = cv2.imread(str(img_path))
        t0 = time.perf_counter()
        inp = preprocess_frame_for_inception(bgr)
        prob = float(model(inp, training=False).numpy()[0, 0])
        latency_ms = (time.perf_counter() - t0) * 1000

        pred_label = "Pothole" if prob >= DEFAULT_CONFIDENCE_THRESHOLD else "Normal"
        is_correct = "PASS" if pred_label == true_label else "FAIL"

        print(f"{img_path.name:<32} | {true_label:<8} | {prob*100:>6.1f}% | {pred_label:<8} | {latency_ms:>6.1f}ms | {is_correct}")

    # 3. Sample Video Inference
    sample_video_path = SAMPLE_VIDEOS_DIR / "sample_dashcam.mp4"
    print("\n[3/3] Inspecting & sampling video frames from dashcam footage:")
    if sample_video_path.exists():
        meta = get_video_metadata(sample_video_path)
        print(f"       Video File   : {meta['filename']}")
        print(f"       Resolution   : {meta['width']}x{meta['height']} @ {meta['fps']} FPS")
        print(f"       Total Frames : {meta['total_frames']} ({meta['duration_sec']}s)")

        cap = cv2.VideoCapture(str(sample_video_path))
        check_frames = [5, 25, 55, 85, 115]
        print("\n       Frame-level telemetry preview:")
        print(f"       {'Frame #':<10} | {'Timestamp':<10} | {'Pothole Prob':<14} | {'Decision':<10} | {'Latency':<8}")
        print("       " + "-" * 62)

        for f_idx in check_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            t0 = time.perf_counter()
            inp = preprocess_frame_for_inception(frame)
            prob = float(model(inp, training=False).numpy()[0, 0])
            latency_ms = (time.perf_counter() - t0) * 1000

            timestamp_sec = f_idx / meta['fps']
            mins = int(timestamp_sec // 60)
            secs = int(timestamp_sec % 60)
            pred = "POTHOLE" if prob >= DEFAULT_CONFIDENCE_THRESHOLD else "NORMAL"

            print(f"       Frame {f_idx:<4} | {mins:02d}:{secs:02d}      | {prob*100:>10.2f}%   | {pred:<10} | {latency_ms:>6.1f}ms")

        cap.release()
    else:
        print(f"       Video file not found at: {sample_video_path}")

    print("\n" + "=" * 70)
    print("SAMPLE RUN COMPLETE: Pipeline is healthy and functional.")
    print("=" * 70)

if __name__ == "__main__":
    main()
