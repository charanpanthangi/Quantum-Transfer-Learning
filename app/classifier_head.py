"""Simple linear classifier head trained on PQC features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .utils import binary_cross_entropy, sigmoid


@dataclass
class LinearClassifierHead:
    """Tiny logistic regression style classifier."""

    weights: np.ndarray
    bias: float

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        """Compute raw logits for a batch of features."""

        return features @ self.weights + self.bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities in [0, 1]."""

        return sigmoid(self.predict_logits(features))

    def predict_class(self, features: np.ndarray) -> np.ndarray:
        """Return hard class predictions (0 or 1)."""

        return (self.predict_proba(features) >= 0.5).astype(int)


def init_head(n_features: int, seed: int | None = 0) -> LinearClassifierHead:
    """Initialize the linear classifier with small random weights."""

    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.2, size=(n_features,), dtype=np.float32)
    bias = float(rng.normal(scale=0.1))
    return LinearClassifierHead(weights=weights, bias=bias)


def train_head(
    features: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_epochs: int = 50,
    seed: int | None = 0,
) -> Tuple[LinearClassifierHead, List[Dict[str, float]]]:
    """Train the head using gradient descent on binary cross-entropy."""

    head = init_head(n_features=features.shape[1], seed=seed)
    history: List[Dict[str, float]] = []

    for epoch in range(n_epochs):
        logits = head.predict_logits(features)
        probs = sigmoid(logits)
        loss = binary_cross_entropy(logits, y)

        # Compute gradients
        grad_logits = probs - y
        grad_w = features.T @ grad_logits / len(y)
        grad_b = float(np.mean(grad_logits))

        # Parameter update
        head.weights -= lr * grad_w
        head.bias -= lr * grad_b

        history.append({"epoch": epoch, "loss": loss})

    return head, history


__all__ = ["LinearClassifierHead", "init_head", "train_head"]
