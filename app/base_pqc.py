"""Base parameterized quantum circuit (PQC) and pretraining utilities.

This module defines a small 2-qubit PQC, initialization helper, and a training
loop for the source task. The trained parameters will later be reused as a
frozen feature extractor for the target task.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Tuple

import numpy as np
import pennylane as qml

from .utils import binary_cross_entropy, sigmoid

# Use a small default device for speed; analytic simulator is enough here
DEV = qml.device("default.qubit", wires=2)


def angle_embedding(x: np.ndarray, wires: List[int]):
    """Encode a 2D point into single-qubit rotations.

    Each dimension is placed into a different axis rotation so the circuit can
    respond to both coordinates.
    """

    qml.RY(x[0], wires=wires[0])
    qml.RZ(x[1], wires=wires[1])


def variational_block(params: np.ndarray):
    """Single variational layer with local rotations and entanglement."""

    for i, wire in enumerate(DEV.wires):
        qml.RX(params[i, 0], wires=wire)
        qml.RY(params[i, 1], wires=wire)
        qml.RZ(params[i, 2], wires=wire)
    # Add entanglement to mix information between qubits
    qml.CNOT(wires=[0, 1])


@functools.partial(qml.qnode, device=DEV)
def base_pqc_qnode(x: np.ndarray, params: np.ndarray) -> float:
    """Quantum node computing an expectation value for binary classification."""

    angle_embedding(x, wires=[0, 1])
    for layer_params in params:
        variational_block(layer_params)
    # Measure PauliZ on the first qubit; result is in [-1, 1]
    return qml.expval(qml.PauliZ(0))


def init_base_pqc_params(n_layers: int = 2, seed: int | None = 0) -> np.ndarray:
    """Initialize PQC parameters from a normal distribution."""

    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.3, size=(n_layers, len(DEV.wires), 3), dtype=np.float32)


def predict_proba(X: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Run the PQC on a batch and return probabilities in [0, 1]."""

    outputs = np.array([base_pqc_qnode(x, params) for x in X])
    probs = (outputs + 1.0) / 2.0
    return probs.astype(np.float32)


def train_base_pqc(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 25,
    lr: float = 0.2,
    n_layers: int = 2,
    seed: int | None = 0,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Train the PQC on the source dataset using gradient descent.

    Args:
        X: Source features (n_samples, 2).
        y: Binary labels.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        n_layers: Number of variational layers.
        seed: Random seed for initialization.

    Returns:
        Trained parameters and a list of history dictionaries containing loss.
    """

    params = init_base_pqc_params(n_layers=n_layers, seed=seed)
    history: List[Dict[str, float]] = []

    # Define differentiable cost using PennyLane gradients
    def cost_fn(current_params):
        probs = predict_proba(X, current_params)
        logits = np.log(probs + 1e-8) - np.log(1 - probs + 1e-8)
        return binary_cross_entropy(logits, y)

    grad_fn = qml.grad(cost_fn)

    for epoch in range(n_epochs):
        grads = grad_fn(params)
        params = params - lr * grads
        loss = cost_fn(params)
        history.append({"epoch": epoch, "loss": loss})

    return params, history


__all__ = [
    "base_pqc_qnode",
    "init_base_pqc_params",
    "predict_proba",
    "train_base_pqc",
]
