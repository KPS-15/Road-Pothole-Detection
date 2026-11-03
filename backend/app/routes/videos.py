"""
Dashcam video processing and streaming routes.
"""

import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse

from backend.app.schemas import (
    VideoUploadResponse,
    VideoProcessRequest,
    VideoStatusResponse,
    VideoResultsResponse,
)
from backend.app.config import (
    UPLOADS_DIR,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    SAMPLE_VIDEOS_DIR,
)
from backend.app.video_service import video_job_manager
from backend.app.model_service import model_service

router = APIRouter(prefix="/api/videos", tags=["Videos"])


@router.post("", response_model=VideoUploadResponse)
async def upload_dashcam_video(file: UploadFile = File(...)):
    filename = file.filename or "dashcam.mp4"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Supported: {sorted(list(ALLOWED_VIDEO_EXTENSIONS))}",
        )

    # Save to upload directory
    target_path = UPLOADS_DIR / f"upload_{Path(filename).stem}_{id(file)}{ext}"
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = target_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video file exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit.",
        )

    try:
        job = video_job_manager.create_upload_job(
            original_filename=filename,
            saved_path=target_path,
            file_size=file_size,
        )
        return VideoUploadResponse(
            job_id=job["job_id"],
            original_filename=job["original_filename"],
            file_size_bytes=job["file_size_bytes"],
            duration_sec=job["duration_sec"],
            total_frames=job["total_frames"],
            fps=job["fps"],
            width=job["width"],
            height=job["height"],
            upload_time=job["upload_time"],
        )
    except Exception as e:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read video metadata: {str(e)}",
        )


@router.post("/{job_id}/process", response_model=VideoStatusResponse)
async def start_video_processing(
    job_id: str,
    req: VideoProcessRequest,
    background_tasks: BackgroundTasks,
):
    job = video_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{job_id}' not found.",
        )

    if not model_service.is_loaded() and not model_service.load_model_if_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model is not available. Please train the model first.",
        )

    if job["status"] == "processing":
        return VideoStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            progress_percentage=job["progress_percentage"],
            current_frame=job["current_frame"],
            total_frames=job["total_frames"],
            processed_frames=job["processed_frames"],
            pothole_count=job["pothole_count"],
            normal_count=job["normal_count"],
        )

    # Schedule async processing in background task
    background_tasks.add_task(
        video_job_manager.run_processing_job,
        job_id=job_id,
        threshold=req.threshold,
        frame_interval=req.frame_interval,
        generate_annotated=req.generate_annotated_video,
    )

    job["status"] = "processing"
    return VideoStatusResponse(
        job_id=job["job_id"],
        status="processing",
        progress_percentage=0.0,
        current_frame=0,
        total_frames=job["total_frames"],
        processed_frames=0,
        pothole_count=0,
        normal_count=0,
    )


@router.get("/{job_id}/status", response_model=VideoStatusResponse)
def get_video_status(job_id: str):
    job = video_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{job_id}' not found.",
        )

    return VideoStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress_percentage=job["progress_percentage"],
        current_frame=job["current_frame"],
        total_frames=job["total_frames"],
        processed_frames=job["processed_frames"],
        pothole_count=job["pothole_count"],
        normal_count=job["normal_count"],
        error=job.get("error"),
    )


@router.get("/{job_id}/results", response_model=VideoResultsResponse)
def get_video_results(job_id: str):
    job = video_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{job_id}' not found.",
        )

    if job["status"] != "completed" or not job.get("results"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video processing not completed. Current status: {job['status']}",
        )

    return VideoResultsResponse(**job["results"])


@router.get("/{job_id}/download")
def download_annotated_video(job_id: str):
    job = video_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{job_id}' not found.",
        )

    annotated_path = job.get("annotated_video_path")
    if not annotated_path or not Path(annotated_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video file not found or not yet generated.",
        )

    return FileResponse(
        path=annotated_path,
        media_type="video/mp4",
        filename=f"annotated_potholes_{job['original_filename']}",
    )
