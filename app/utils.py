"""Utility helpers for metrics and simple operations."""

from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute simple classification accuracy.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Predicted labels (0/1).

    Returns:
        Accuracy in [0, 1].
    """

    return float(np.mean(y_true == y_pred))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    # Clip to avoid overflow in exp
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Compute binary cross-entropy given logits.

    Args:
        logits: Model outputs before sigmoid.
        y_true: Ground truth labels (0/1).

    Returns:
        Scalar loss value.
    """

    probs = sigmoid(logits)
    eps = 1e-8
    probs = np.clip(probs, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
    return float(loss)


__all__ = ["accuracy_score", "sigmoid", "binary_cross_entropy"]
