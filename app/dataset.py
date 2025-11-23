"""Dataset utilities for the Quantum Transfer Learning demo.

This module creates small 2D synthetic datasets for source and target tasks.
The goal is to keep the data simple, fast, and easily visualized while still
showing a small domain shift between source and target tasks.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split as sk_train_test_split


def generate_source_dataset(n_samples: int = 200, random_state: int | None = 0):
    """Generate a simple two-blob dataset for the source task.

    The classes are separable but overlap slightly so the PQC must learn a
    meaningful decision boundary.

    Args:
        n_samples: Number of total samples (evenly split across classes).
        random_state: Optional seed for reproducibility.

    Returns:
        Tuple of (X, y) where X has shape (n_samples, 2) and y contains labels
        in {0, 1}.
    """

    X, y = make_blobs(
        n_samples=n_samples,
        centers=[(-1.0, -1.0), (1.0, 1.0)],
        cluster_std=[0.6, 0.6],
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.int64)


def generate_target_dataset(n_samples: int = 200, random_state: int | None = 1):
    """Generate a slightly shifted moons dataset for the target task.

    The dataset differs from the source task to create a small domain shift
    where transfer learning can help.

    Args:
        n_samples: Number of total samples.
        random_state: Optional seed for reproducibility.

    Returns:
        Tuple of (X, y) similar to :func:`generate_source_dataset`.
    """

    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    # Shift and scale slightly to make it related but not identical
    X = X * 1.2 + np.array([0.3, -0.1])
    return X.astype(np.float32), y.astype(np.int64)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int | None = 42
):
    """Split the dataset into train and test sets.

    Args:
        X: Feature matrix of shape (n_samples, 2).
        y: Labels vector.
        test_size: Fraction to include in the test split.
        random_state: Optional seed.

    Returns:
        X_train, X_test, y_train, y_test
    """

    return sk_train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


__all__ = [
    "generate_source_dataset",
    "generate_target_dataset",
    "train_test_split",
]
