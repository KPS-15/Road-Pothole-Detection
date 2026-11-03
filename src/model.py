"""
InceptionV3 Transfer Learning Architecture for Road Pothole Detection.
Implements two-phase training (Frozen Backbone Head Training + Upper Layer Fine-Tuning),
custom callbacks, evaluation metrics, and model persistence.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications.inception_v3 import InceptionV3

from src.config import (
    INPUT_SHAPE,
    DROPOUT_RATE,
    HEAD_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    MIN_LR,
    BEST_MODEL_PATH,
    TRAINING_CURVES_PATH,
    MODEL_CONFIG_PATH,
    TRAINING_LOG_CSV,
)


def get_available_device_info() -> Dict[str, Any]:
    """Inspect and report physical hardware acceleration (GPU vs CPU)."""
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    device_info = {
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "gpu_devices": [gpu.name for gpu in gpus],
        "cpu_count": len(cpus),
        "tensorflow_version": tf.__version__,
    }
    return device_info


def build_inceptionv3_pothole_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = HEAD_LEARNING_RATE,
    weights: str = "imagenet",
) -> tf.keras.Model:
    """
    Constructs the InceptionV3 Transfer Learning Model:
    1. Pretrained InceptionV3 backbone (frozen).
    2. GlobalAveragePooling2D to compress feature maps into a vector.
    3. Dropout for regularization.
    4. Dense(1, activation='sigmoid') for binary pothole probability.
    """
    # 1. Base Pretrained Backbone
    base_model = InceptionV3(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )
    # Freeze the feature extractor backbone for Phase 1
    base_model.trainable = False

    # 2. Functional Architecture
    inputs = tf.keras.Input(shape=input_shape, name="dashcam_frame_input")
    x = base_model(inputs, training=False)  # Ensure batchnorm stays in inference mode
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout_regularization")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pothole_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="InceptionV3_Pothole_Detector")

    # 3. Compile Model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def prepare_for_fine_tuning(
    model: tf.keras.Model,
    unfreeze_from_layer_name: Optional[str] = "conv2d_85",
    unfreeze_last_n_layers: int = 50,
    fine_tune_lr: float = FINE_TUNE_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Unfreezes upper layers of the InceptionV3 backbone while keeping lower feature
    extractors and BatchNormalization layers frozen in inference mode.
    Recompiles with a reduced learning rate to prevent catastrophic forgetting.
    """
    # Find the InceptionV3 backbone sub-layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "inception_v3" in layer.name:
            base_model = layer
            break

    if base_model is None:
        base_model = model

    base_model.trainable = True

    layer_names = [l.name for l in base_model.layers]
    
    # If the exact layer name is found in this instance of InceptionV3, use it as cutoff
    if unfreeze_from_layer_name and unfreeze_from_layer_name in layer_names:
        cutoff_idx = layer_names.index(unfreeze_from_layer_name)
    else:
        # Fallback for repeated model instantiations where layer names auto-increment
        cutoff_idx = max(0, len(base_model.layers) - unfreeze_last_n_layers)

    for idx, layer in enumerate(base_model.layers):
        is_upper_layer = (idx >= cutoff_idx)
        # Keep BatchNormalization frozen to preserve pretrained running mean & variance
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = is_upper_layer

    optimizer = optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_training_callbacks(
    checkpoint_filepath: Path = BEST_MODEL_PATH,
    log_csv_filepath: Path = TRAINING_LOG_CSV,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    reduce_lr_patience: int = REDUCE_LR_PATIENCE,
    reduce_lr_factor: float = REDUCE_LR_FACTOR,
    min_lr: float = MIN_LR,
) -> List[callbacks.Callback]:
    """Constructs comprehensive training callbacks."""
    Path(checkpoint_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(log_csv_filepath).parent.mkdir(parents=True, exist_ok=True)

    cb_list = [
        # Checkpoint: Save best model based on validation loss
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_filepath),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        # Early Stopping: Prevent overfitting on validation plateau
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR On Plateau: Decay learning rate when validation loss stops improving
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        # CSV Logger: Persist epoch-by-epoch loss & metric history
        callbacks.CSVLogger(
            filename=str(log_csv_filepath),
            separator=",",
            append=False,
        ),
    ]
    return cb_list


def plot_and_save_training_history(
    history_records: List[Dict[str, List[float]]],
    output_path: Path = TRAINING_CURVES_PATH,
) -> None:
    """
    Combines Phase 1 and Phase 2 history and renders dual-panel training curves:
    Panel 1: Accuracy (Train vs Val)
    Panel 2: Binary Cross-Entropy Loss (Train vs Val)
    """
    # Merge histories
    acc, val_acc = [], []
    loss, val_loss = [], []
    for h in history_records:
        acc.extend(h.get("accuracy", []))
        val_acc.extend(h.get("val_accuracy", []))
        loss.extend(h.get("loss", []))
        val_loss.extend(h.get("val_loss", []))

    if not acc:
        return

    epochs = range(1, len(acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Accuracy Curve
    ax1.plot(epochs, acc, label="Training Accuracy", color="#1976D2", linewidth=2, marker="o", markersize=3)
    ax1.plot(epochs, val_acc, label="Validation Accuracy", color="#FF9800", linewidth=2, marker="s", markersize=3)
    ax1.set_title("Model Accuracy: Training vs Validation", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=10, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Loss Curve
    ax2.plot(epochs, loss, label="Training Loss", color="#1976D2", linewidth=2, marker="o", markersize=3)
    ax2.plot(epochs, val_loss, label="Validation Loss", color="#E53935", linewidth=2, marker="s", markersize=3)
    ax2.set_title("Binary Cross-Entropy Loss: Training vs Validation", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_model_configuration(
    settings: Dict[str, Any],
    output_path: Path = MODEL_CONFIG_PATH,
) -> None:
    """Save model configuration and training hyperparameters to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)
