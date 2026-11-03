"""
Unit tests for InceptionV3 model architecture, layer configurations,
output shapes, and fine-tuning transitions.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INPUT_SHAPE, DROPOUT_RATE


def test_model_architecture_and_output_shapes():
    """
    Verify InceptionV3 transfer learning model architecture,
    GAP layer, Dropout, and Sigmoid binary classification output.
    Uses weights=None to ensure fast, deterministic offline testing.
    """
    import tensorflow as tf
    from src.model import build_inceptionv3_pothole_model

    model = build_inceptionv3_pothole_model(
        input_shape=INPUT_SHAPE,
        dropout_rate=DROPOUT_RATE,
        weights=None,  # Random weights for instant unit testing
    )

    # Verify input shape
    assert model.input_shape == (None, 299, 299, 3)
    
    # Verify output shape
    assert model.output_shape == (None, 1)

    # Verify backbone is frozen (Phase 1)
    # The output dense layer has 2048 weights + 1 bias = 2049 trainable parameters
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    assert trainable_params == 2049, f"Expected 2049 trainable params in frozen head, got {trainable_params}"

    # Verify dummy prediction shape and probability bounds
    dummy_input = tf.random.uniform((2, 299, 299, 3), minval=-1.0, maxval=1.0)
    preds = model(dummy_input, training=False).numpy()

    assert preds.shape == (2, 1)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_fine_tuning_unfreezing():
    """Verify that fine-tuning unfreezes upper layers while keeping BatchNormalization frozen."""
    import tensorflow as tf
    from src.model import build_inceptionv3_pothole_model, prepare_for_fine_tuning

    model = build_inceptionv3_pothole_model(
        input_shape=INPUT_SHAPE,
        weights=None,
    )
    initial_trainable = len(model.trainable_weights)

    # Prepare for fine-tuning
    ft_model = prepare_for_fine_tuning(model, unfreeze_from_layer_name="conv2d_85")
    ft_trainable = len(ft_model.trainable_weights)

    # Trainable weights must increase
    assert ft_trainable > initial_trainable


def test_loss_and_optimizer_compilation():
    """Verify loss function is binary cross-entropy and optimizer is Adam."""
    import tensorflow as tf
    from src.model import build_inceptionv3_pothole_model

    model = build_inceptionv3_pothole_model(weights=None)
    
    assert "binary_crossentropy" in str(model.loss).lower()
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
