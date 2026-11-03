"""
Pydantic schemas for API request validation and structured response serialization.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    device: str
    gpu_available: bool


class ModelStatusResponse(BaseModel):
    is_available: bool
    is_loaded: bool
    model_path: str
    architecture: str = "InceptionV3"
    input_shape: List[int]
    total_params: Optional[int] = None
    config: Optional[Dict[str, Any]] = None


class ImagePredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    pothole_probability: float
    confidence_percentage: float
    is_hazard: bool
    inference_time_ms: float
    threshold_applied: float


class VideoUploadResponse(BaseModel):
    job_id: str
    original_filename: str
    file_size_bytes: int
    duration_sec: float
    total_frames: int
    fps: float
    width: int
    height: int
    upload_time: str


class VideoProcessRequest(BaseModel):
    threshold: float = Field(default=0.50, ge=0.05, le=0.95, description="Alert probability threshold")
    frame_interval: int = Field(default=3, ge=1, le=30, description="Process every Nth frame")
    generate_annotated_video: bool = Field(default=True, description="Render overlay and generate MP4")


class FramePrediction(BaseModel):
    frame_number: int
    timestamp_sec: float
    predicted_class: str
    pothole_probability: float
    is_pothole: bool
    inference_time_ms: float
    thumbnail_data_url: Optional[str] = None


class VideoStatusResponse(BaseModel):
    job_id: str
    status: str  # "uploaded", "processing", "completed", "failed"
    progress_percentage: float
    current_frame: int
    total_frames: int
    processed_frames: int
    pothole_count: int
    normal_count: int
    error: Optional[str] = None


class VideoResultsResponse(BaseModel):
    job_id: str
    status: str
    original_filename: str
    total_video_frames: int
    evaluated_frames_count: int
    pothole_frames_count: int
    normal_frames_count: int
    pothole_percentage: float
    average_latency_ms: float
    estimated_fps: float
    total_processing_time_sec: float
    has_annotated_video: bool
    frame_timeline: List[FramePrediction]
    top_pothole_frames: List[FramePrediction]


class MetricsResponse(BaseModel):
    has_results: bool
    summary: Optional[Dict[str, Any]] = None
    classification_report: Optional[str] = None
    dataset_manifest: Optional[Dict[str, Any]] = None
