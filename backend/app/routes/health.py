"""
Health and model status routes.
"""

import json
from pathlib import Path
from fastapi import APIRouter
import tensorflow as tf

from backend.app.schemas import HealthResponse, ModelStatusResponse, MetricsResponse
from backend.app.model_service import model_service
from backend.app.config import (
    METRICS_SUMMARY_PATH,
    CLASSIFICATION_REPORT_PATH,
    RESULTS_DIR,
)

router = APIRouter(prefix="/api", tags=["System & Health"])


@router.get("/health", response_model=HealthResponse)
def get_health():
    is_gpu = model_service._use_gpu
    device_desc = "NVIDIA GeForce RTX 5050 (DirectML GPU)" if is_gpu else "CPU (Multi-Threaded)"
    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=device_desc,
        gpu_available=is_gpu,
    )


@router.get("/model/status", response_model=ModelStatusResponse)
def get_model_status():
    status = model_service.get_status()
    return ModelStatusResponse(
        is_available=status["is_available"],
        is_loaded=status["is_loaded"],
        model_path=status["model_path"],
        architecture=status["architecture"],
        input_shape=status["input_shape"],
        total_params=status["total_params"],
        config=status["config"],
    )


@router.get("/evaluation/metrics", response_model=MetricsResponse)
def get_evaluation_metrics():
    summary_data = None
    report_text = None
    manifest_data = None

    if METRICS_SUMMARY_PATH.exists():
        try:
            with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    if CLASSIFICATION_REPORT_PATH.exists():
        try:
            with open(CLASSIFICATION_REPORT_PATH, "r", encoding="utf-8") as f:
                report_text = f.read()
        except Exception:
            pass

    manifest_path = RESULTS_DIR / "dataset_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    has_results = summary_data is not None
    return MetricsResponse(
        has_results=has_results,
        summary=summary_data,
        classification_report=report_text,
        dataset_manifest=manifest_data,
    )
