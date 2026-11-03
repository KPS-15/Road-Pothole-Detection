"""
Dashcam Video Processing Pipeline for Road Pothole Detection.
Provides frame extraction, batch/frame-level preprocessing matching InceptionV3,
HUD telemetry overlay, video annotation, and robust codec fallback handling.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Generator
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_SAMPLE_INTERVAL,
    POSITIVE_CLASS,
    NEGATIVE_CLASS,
    VALID_VIDEO_EXTENSIONS,
)


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Inspect video properties (resolution, total frames, FPS, duration)."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (total_frames / fps) if fps > 0 else 0.0
    cap.release()

    return {
        "filepath": str(video_path),
        "filename": video_path.name,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
    }


def preprocess_frame_for_inception(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Converts a single BGR OpenCV frame into preprocessed InceptionV3 input:
    1. BGR -> RGB color conversion.
    2. Resizing to 299x299.
    3. Official InceptionV3 normalization to [-1, 1].
    4. Expands dims to (1, 299, 299, 3).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    # Ensure float32 for model input
    float_img = resized.astype(np.float32)
    # InceptionV3 preprocessing scales [0, 255] to [-1, 1]
    normalized = preprocess_input(float_img)
    return np.expand_dims(normalized, axis=0)


def draw_hud_overlay(
    frame: np.ndarray,
    probability: float,
    threshold: float,
    frame_idx: int,
    total_frames: int,
    fps: float,
    latency_ms: float,
) -> np.ndarray:
    """
    Renders an informative, high-visibility HUD banner across the frame.
    Explicitly clarifies frame-level classification confidence without fake bounding boxes.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    is_pothole = probability >= threshold
    timestamp_sec = frame_idx / fps if fps > 0 else 0.0
    mins = int(timestamp_sec // 60)
    secs = int(timestamp_sec % 60)

    # Styling colors (BGR format for OpenCV)
    # Pothole: Crimson Red (0, 30, 215)
    # Normal: Emerald Green (40, 160, 40)
    banner_color = (25, 25, 215) if is_pothole else (40, 160, 40)
    text_color = (255, 255, 255)

    # Top Banner Header
    banner_height = max(70, int(h * 0.12))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    # Add semi-transparent blend
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

    # Alert Headline
    status_text = (
        f"POTHOLE DETECTED [ALERT]"
        if is_pothole
        else "NORMAL ROAD SURFACE"
    )
    cv2.putText(
        annotated,
        status_text,
        (20, int(banner_height * 0.45)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        text_color,
        2,
        cv2.LINE_AA,
    )

    # Probability & Threshold Telemetry Sub-header
    detail_text = (
        f"Confidence: {probability * 100:.1f}% | "
        f"Threshold: {threshold * 100:.0f}% | "
        f"Time: {mins:02d}:{secs:02d} | "
        f"Frame: {frame_idx}/{total_frames} | "
        f"Latency: {latency_ms:.1f}ms"
    )
    cv2.putText(
        annotated,
        detail_text,
        (20, int(banner_height * 0.82)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # Bottom Scope Notice: Prevents user mistaking frame classification for bounding boxes
    bottom_height = max(30, int(h * 0.05))
    overlay_bottom = annotated.copy()
    cv2.rectangle(overlay_bottom, (0, h - bottom_height), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay_bottom, 0.75, annotated, 0.25, 0, annotated)
    scope_text = "Notice: InceptionV3 Frame-Level Road Classification (Not Pixel Bounding-Box Detection)"
    cv2.putText(
        annotated,
        scope_text,
        (15, h - int(bottom_height * 0.3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return annotated


def process_dashcam_video(
    video_path: Path,
    model: tf.keras.Model,
    output_path: Optional[Path] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    sample_interval: int = DEFAULT_FRAME_SAMPLE_INTERVAL,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end dashcam video inference:
    1. Opens input video stream.
    2. Samples frames according to sample_interval.
    3. Runs InceptionV3 model inference.
    4. Annotates frames with HUD banners.
    5. Writes annotated video file using robust codec fallbacks.
    6. Returns complete telemetry log and prediction metrics.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open dashcam video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Initialize VideoWriter if output requested
    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Try H.264 mp4v or fallback codecs
        codecs_to_try = [
            ("mp4v", ".mp4"),
            ("avc1", ".mp4"),
            ("XVID", ".avi"),
        ]
        for fourcc_str, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out_candidate = output_path.with_suffix(ext)
                writer = cv2.VideoWriter(str(out_candidate), fourcc, fps / sample_interval, (width, height))
                if writer.isOpened():
                    output_path = out_candidate
                    break
            except Exception:
                writer = None

    frame_idx = 0
    processed_count = 0
    pothole_positive_count = 0
    normal_count = 0
    telemetry_records = []
    latencies = []
    top_pothole_frames: List[Tuple[float, np.ndarray, int]] = []

    last_prob = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % sample_interval == 0:
                # Preprocess & Infer
                t0 = time.perf_counter()
                preprocessed = preprocess_frame_for_inception(frame)
                pred_prob = float(model(preprocessed, training=False).numpy()[0, 0])
                latency_ms = (time.perf_counter() - t0) * 1000.0

                last_prob = pred_prob
                processed_count += 1
                latencies.append(latency_ms)

                is_pothole = pred_prob >= threshold
                if is_pothole:
                    pothole_positive_count += 1
                    # Track top high-confidence frames for summary gallery
                    if len(top_pothole_frames) < 6:
                        top_pothole_frames.append((pred_prob, frame.copy(), frame_idx))
                    else:
                        top_pothole_frames.sort(key=lambda x: x[0])
                        if pred_prob > top_pothole_frames[0][0]:
                            top_pothole_frames[0] = (pred_prob, frame.copy(), frame_idx)
                else:
                    normal_count += 1

                record = {
                    "frame_index": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "probability": round(pred_prob, 4),
                    "is_pothole": is_pothole,
                    "latency_ms": round(latency_ms, 2),
                }
                telemetry_records.append(record)

                if progress_callback:
                    progress_callback(frame_idx, total_frames, record)

                if writer and writer.isOpened():
                    annotated_frame = draw_hud_overlay(
                        frame, pred_prob, threshold, frame_idx, total_frames, fps, latency_ms
                    )
                    writer.write(annotated_frame)
    finally:
        cap.release()
        if writer and writer.isOpened():
            writer.release()

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    approx_fps = float(1000.0 / avg_latency) if avg_latency > 0 else 0.0

    summary = {
        "video_path": str(video_path),
        "output_path": str(output_path) if (output_path and output_path.exists()) else None,
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "pothole_positive_frames": pothole_positive_count,
        "normal_frames": normal_count,
        "pothole_ratio_pct": round((pothole_positive_count / processed_count * 100), 2) if processed_count > 0 else 0.0,
        "mean_latency_ms": round(avg_latency, 2),
        "approx_processing_fps": round(approx_fps, 2),
        "telemetry_records": telemetry_records,
        "top_pothole_frames": sorted(top_pothole_frames, key=lambda x: x[0], reverse=True),
    }
    return summary
