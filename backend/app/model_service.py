"""
Model service for loading and executing InceptionV3 transfer learning inference.
Guarantees thread-safe singleton model loading and consistent preprocessing.
"""

import io
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

from backend.app.config import (
    BEST_MODEL_PATH,
    MODEL_CONFIG_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    INPUT_SHAPE,
    DEFAULT_THRESHOLD,
)


class ModelService:
    def __init__(self):
        self._model: Optional[tf.keras.Model] = None
        self._ort_session: Any = None
        self._input_name: Optional[str] = None
        self._use_gpu: bool = False
        self._model_config: Optional[Dict[str, Any]] = None
        self.load_model_if_available()

    def is_available(self) -> bool:
        onnx_path = BEST_MODEL_PATH.with_suffix(".onnx")
        return onnx_path.exists() or BEST_MODEL_PATH.exists()

    def is_loaded(self) -> bool:
        return self._ort_session is not None or self._model is not None

    def load_model_if_available(self) -> bool:
        if self.is_loaded():
            return True

        onnx_path = BEST_MODEL_PATH.with_suffix(".onnx")
        try:
            import onnxruntime as ort
            if onnx_path.exists() and "DmlExecutionProvider" in ort.get_available_providers():
                print(f"[ModelService] Loading ONNX model onto GPU via DirectML from {onnx_path}...")
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                self._ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
                self._input_name = self._ort_session.get_inputs()[0].name
                self._use_gpu = True
                print("[ModelService] [HARDWARE ACCELERATION ACTIVE] Successfully loaded onto NVIDIA GeForce RTX 5050 (DirectML GPU).")
                if MODEL_CONFIG_PATH.exists():
                    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                        self._model_config = json.load(f)
                return True
        except Exception as e:
            print(f"[ModelService] DirectML loading fallback note: {e}")

        if not BEST_MODEL_PATH.exists():
            return False

        try:
            print(f"[ModelService] Loading trained InceptionV3 model from {BEST_MODEL_PATH}...")
            self._model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
            if MODEL_CONFIG_PATH.exists():
                with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                    self._model_config = json.load(f)
            print("[ModelService] Successfully loaded trained InceptionV3 model.")
            return True
        except Exception as e:
            print(f"[ModelService] Error loading model: {e}")
            self._model = None
            return False

    def get_status(self) -> Dict[str, Any]:
        available = self.is_available()
        loaded = self.is_loaded()
        total_params = None
        if loaded and self._model is not None:
            try:
                total_params = self._model.count_params()
            except Exception:
                pass

        return {
            "is_available": available,
            "is_loaded": loaded,
            "model_path": str(BEST_MODEL_PATH),
            "architecture": "InceptionV3",
            "input_shape": list(INPUT_SHAPE),
            "total_params": total_params,
            "config": self._model_config,
        }

    def preprocess_pil_image(self, img: Image.Image) -> np.ndarray:
        """Preprocesses a PIL Image into (1, 299, 299, 3) normalized [-1, 1]."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        norm_arr = preprocess_input(arr)
        return norm_arr

    def preprocess_cv2_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Preprocesses an OpenCV BGR frame into (1, 299, 299, 3) normalized [-1, 1]."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        arr = np.expand_dims(resized.astype(np.float32), axis=0)
        norm_arr = preprocess_input(arr)
        return norm_arr

    def predict_image(
        self, image_bytes: bytes, filename: str, threshold: float = DEFAULT_THRESHOLD
    ) -> Dict[str, Any]:
        if not self.is_loaded():
            if not self.load_model_if_available():
                raise RuntimeError(
                    f"Model checkpoint not found at '{BEST_MODEL_PATH}'. "
                    "Please train the model using 'python train.py' before running inference."
                )

        img = Image.open(io.BytesIO(image_bytes))
        inp = self.preprocess_pil_image(img)

        t0 = time.perf_counter()
        if self._use_gpu and self._ort_session is not None:
            inp_data = inp.astype(np.float32)
            raw_pred = self._ort_session.run(None, {self._input_name: inp_data})[0]
        else:
            raw_pred = self._model(inp, training=False).numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pothole_prob = float(raw_pred[0, 0])
        is_hazard = pothole_prob >= threshold
        predicted_class = "Pothole" if is_hazard else "Normal"
        confidence_pct = (pothole_prob if is_hazard else (1.0 - pothole_prob)) * 100.0

        return {
            "filename": filename,
            "predicted_class": predicted_class,
            "pothole_probability": round(pothole_prob, 4),
            "confidence_percentage": round(confidence_pct, 2),
            "is_hazard": is_hazard,
            "inference_time_ms": round(latency_ms, 2),
            "threshold_applied": threshold,
            "device": "NVIDIA GeForce RTX 5050 (DirectML GPU)" if self._use_gpu else "CPU",
        }

    def predict_frame(
        self, bgr_frame: np.ndarray, threshold: float = DEFAULT_THRESHOLD
    ) -> Tuple[float, str, bool, float]:
        """Runs fast inference on a single CV2 BGR frame."""
        if not self.is_loaded():
            if not self.load_model_if_available():
                raise RuntimeError("Model is not loaded.")

        inp = self.preprocess_cv2_frame(bgr_frame)
        t0 = time.perf_counter()
        if self._use_gpu and self._ort_session is not None:
            inp_data = inp.astype(np.float32)
            raw_pred = self._ort_session.run(None, {self._input_name: inp_data})[0]
        else:
            raw_pred = self._model(inp, training=False).numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob = float(raw_pred[0, 0])
        is_hazard = prob >= threshold
        cls_name = "Pothole" if is_hazard else "Normal"
        return prob, cls_name, is_hazard, latency_ms


# Global singleton instance
model_service = ModelService()
