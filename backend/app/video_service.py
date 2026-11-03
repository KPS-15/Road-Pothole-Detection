"""
Video processing service for dashcam footage.
Manages job queue, asynchronous frame-by-frame inference, HUD overlay generation,
and downloadable annotated video generation.
"""

import os
import uuid
import time
import base64
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2
import numpy as np

from backend.app.config import (
    UPLOADS_DIR,
    PROCESSED_VIDEOS_DIR,
    DEFAULT_THRESHOLD,
    DEFAULT_FRAME_INTERVAL,
)
from backend.app.model_service import model_service
from src.video_utils import draw_hud_overlay


class VideoJobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_upload_job(
        self, original_filename: str, saved_path: Path, file_size: int
    ) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())[:8]

        cap = cv2.VideoCapture(str(saved_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open or decode video file: {original_filename}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0
        cap.release()

        job_data = {
            "job_id": job_id,
            "original_filename": original_filename,
            "video_path": str(saved_path),
            "file_size_bytes": file_size,
            "duration_sec": duration_sec,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "status": "uploaded",  # uploaded, processing, completed, failed
            "progress_percentage": 0.0,
            "current_frame": 0,
            "processed_frames": 0,
            "pothole_count": 0,
            "normal_count": 0,
            "error": None,
            "results": None,
            "annotated_video_path": None,
            "upload_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self.jobs[job_id] = job_data
        return job_data

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    async def run_processing_job(
        self,
        job_id: str,
        threshold: float = DEFAULT_THRESHOLD,
        frame_interval: int = DEFAULT_FRAME_INTERVAL,
        generate_annotated: bool = True,
    ) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return

        job["status"] = "processing"
        video_path = job["video_path"]
        total_frames = job["total_frames"] or 1
        fps = job["fps"] or 25.0
        width = job["width"]
        height = job["height"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            job["status"] = "failed"
            job["error"] = "Failed to open video for processing."
            return

        annotated_path = None
        writer = None
        if generate_annotated:
            out_filename = f"annotated_{job_id}.mp4"
            annotated_path = PROCESSED_VIDEOS_DIR / out_filename
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(
                str(annotated_path),
                fourcc,
                fps / frame_interval,
                (width, height),
            )
            # Fallback codec if avc1 is unavailable on Windows
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(annotated_path),
                    fourcc,
                    fps / frame_interval,
                    (width, height),
                )

        current_frame_idx = 0
        processed_count = 0
        pothole_count = 0
        normal_count = 0
        frame_predictions: List[Dict[str, Any]] = []
        top_pothole_frames: List[Dict[str, Any]] = []
        latencies: List[float] = []

        start_time = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_idx += 1

                if current_frame_idx % frame_interval == 0:
                    prob, cls_name, is_hazard, lat_ms = model_service.predict_frame(
                        frame, threshold=threshold
                    )
                    latencies.append(lat_ms)
                    processed_count += 1

                    if is_hazard:
                        pothole_count += 1
                    else:
                        normal_count += 1

                    timestamp_sec = round(current_frame_idx / fps, 2)
                    frame_record = {
                        "frame_number": current_frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "predicted_class": cls_name,
                        "pothole_probability": round(prob, 4),
                        "is_pothole": is_hazard,
                        "inference_time_ms": round(lat_ms, 2),
                    }
                    frame_predictions.append(frame_record)

                    # Annotate frame
                    annotated_frame = draw_hud_overlay(
                        frame,
                        prob,
                        threshold,
                        current_frame_idx,
                        total_frames,
                        fps,
                        lat_ms,
                    )

                    if writer and writer.isOpened():
                        writer.write(annotated_frame)

                    # Save thumbnail for top potholes
                    if is_hazard and len(top_pothole_frames) < 12:
                        small_frame = cv2.resize(annotated_frame, (320, 180))
                        _, buffer = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                        b64_str = base64.b64encode(buffer).decode("utf-8")
                        top_pothole_frames.append({
                            **frame_record,
                            "thumbnail_data_url": f"data:image/jpeg;base64,{b64_str}",
                        })

                    # Update status
                    job["current_frame"] = current_frame_idx
                    job["processed_frames"] = processed_count
                    job["pothole_count"] = pothole_count
                    job["normal_count"] = normal_count
                    job["progress_percentage"] = round(
                        min(99.0, (current_frame_idx / total_frames) * 100.0), 1
                    )

                    # Yield control periodically to allow event loop handling
                    if processed_count % 5 == 0:
                        await asyncio.sleep(0.001)

            cap.release()
            if writer and writer.isOpened():
                writer.release()

            total_elapsed = round(time.perf_counter() - start_time, 2)
            avg_lat = round(float(np.mean(latencies)), 2) if latencies else 0.0
            est_fps = round((1000.0 / avg_lat), 1) if avg_lat > 0 else 0.0
            pothole_ratio = (
                round((pothole_count / processed_count) * 100.0, 1) if processed_count > 0 else 0.0
            )

            # Sort top pothole frames by highest probability
            top_pothole_frames.sort(key=lambda x: x["pothole_probability"], reverse=True)

            job["status"] = "completed"
            job["progress_percentage"] = 100.0
            job["annotated_video_path"] = str(annotated_path) if (annotated_path and annotated_path.exists()) else None
            job["results"] = {
                "job_id": job_id,
                "status": "completed",
                "original_filename": job["original_filename"],
                "total_video_frames": total_frames,
                "evaluated_frames_count": processed_count,
                "pothole_frames_count": pothole_count,
                "normal_frames_count": normal_count,
                "pothole_percentage": pothole_ratio,
                "average_latency_ms": avg_lat,
                "estimated_fps": est_fps,
                "total_processing_time_sec": total_elapsed,
                "has_annotated_video": job["annotated_video_path"] is not None,
                "frame_timeline": frame_predictions,
                "top_pothole_frames": top_pothole_frames[:9],
            }

        except Exception as e:
            cap.release()
            if writer and writer.isOpened():
                writer.release()
            job["status"] = "failed"
            job["error"] = str(e)


video_job_manager = VideoJobManager()
