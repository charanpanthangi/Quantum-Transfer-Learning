"""Frozen PQC feature extraction utilities."""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .base_pqc import DEV, angle_embedding, variational_block


def build_feature_qnode(n_observables: int = 3):
    """Create a QNode that measures multiple observables for richer features."""

    observables = []
    if n_observables >= 1:
        observables.append(qml.PauliZ(0))
    if n_observables >= 2:
        observables.append(qml.PauliZ(1))
    if n_observables >= 3:
        observables.append(qml.PauliZ(0) @ qml.PauliZ(1))

    @qml.qnode(device=DEV)
    def qnode(x, params):
        angle_embedding(x, wires=[0, 1])
        for layer_params in params:
            variational_block(layer_params)
        return [qml.expval(obs) for obs in observables]

    return qnode


def extract_features(X: np.ndarray, params_base: np.ndarray, n_observables: int = 3) -> np.ndarray:
    """Extract features from inputs using a frozen PQC.

    Args:
        X: Input points (n_samples, 2).
        params_base: Pretrained PQC parameters (kept fixed).
        n_observables: Number of observables to measure for each sample.

    Returns:
        Feature matrix of shape (n_samples, n_observables).
    """

    qnode = build_feature_qnode(n_observables=n_observables)
    features = [qnode(x, params_base) for x in X]
    return np.array(features, dtype=np.float32)


__all__ = ["extract_features", "build_feature_qnode"]
