"""Transfer learning experiments combining PQC backbone and classifier head."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pennylane as qml

from .base_pqc import train_base_pqc
from .classifier_head import LinearClassifierHead, init_head, train_head
from .feature_extractor import extract_features
from .utils import accuracy_score, binary_cross_entropy, sigmoid


def run_frozen_transfer_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_params: np.ndarray,
    head_lr: float = 0.1,
    head_epochs: int = 50,
) -> Tuple[LinearClassifierHead, Dict[str, float], Dict[str, list]]:
    """Train only the classifier head using frozen PQC features."""

    train_features = extract_features(X_train, base_params)
    test_features = extract_features(X_test, base_params)

    head, history = train_head(train_features, y_train, lr=head_lr, n_epochs=head_epochs)
    test_preds = head.predict_class(test_features)
    test_logits = head.predict_logits(test_features)
    test_loss = binary_cross_entropy(test_logits, y_test)
    test_acc = accuracy_score(y_test, test_preds)

    metrics = {"loss": test_loss, "accuracy": test_acc}
    histories = {"head": history}
    return head, metrics, histories


def run_finetune_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    init_params: np.ndarray,
    head_lr: float = 0.1,
    pqc_lr: float = 0.1,
    n_epochs: int = 30,
) -> Tuple[np.ndarray, LinearClassifierHead, Dict[str, float], Dict[str, list]]:
    """Jointly train PQC parameters and classifier head."""

    params = np.copy(init_params)
    head = init_head(n_features=3, seed=0)

    history = []

    def joint_cost(current_params, current_head: LinearClassifierHead):
        features = extract_features(X_train, current_params)
        logits = features @ current_head.weights + current_head.bias
        return binary_cross_entropy(logits, y_train)

    grad_params_fn = lambda p, h: qml.grad(lambda par: joint_cost(par, h))(p)

    for epoch in range(n_epochs):
        # PQC gradient
        grads_params = grad_params_fn(params, head)
        params = params - pqc_lr * grads_params

        # Update head using fresh features
        train_features = extract_features(X_train, params)
        logits = head.predict_logits(train_features)
        probs = sigmoid(logits)
        loss = binary_cross_entropy(logits, y_train)
        grad_logits = probs - y_train
        grad_w = train_features.T @ grad_logits / len(y_train)
        grad_b = float(np.mean(grad_logits))
        head.weights -= head_lr * grad_w
        head.bias -= head_lr * grad_b

        # Track metrics on test set
        test_features = extract_features(X_test, params)
        test_logits = head.predict_logits(test_features)
        test_preds = (sigmoid(test_logits) >= 0.5).astype(int)
        test_loss = binary_cross_entropy(test_logits, y_test)
        test_acc = accuracy_score(y_test, test_preds)
        history.append({
            "epoch": epoch,
            "train_loss": float(loss),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        })

    metrics = {"loss": history[-1]["test_loss"], "accuracy": history[-1]["test_accuracy"]}
    histories = {"joint": history}
    return params, head, metrics, histories


def run_all_experiments(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    n_base_epochs: int = 15,
    n_target_epochs: int = 25,
) -> Dict[str, Dict[str, float]]:
    """Pretrain base PQC then run frozen and fine-tuning experiments."""

    base_params, base_history = train_base_pqc(X_source, y_source, n_epochs=n_base_epochs)

    # Split target data
    split_idx = int(0.75 * len(X_target))
    X_train, X_test = X_target[:split_idx], X_target[split_idx:]
    y_train, y_test = y_target[:split_idx], y_target[split_idx:]

    _, frozen_metrics, frozen_history = run_frozen_transfer_experiment(
        X_train, y_train, X_test, y_test, base_params, head_epochs=n_target_epochs
    )

    finetune_params, finetune_head, finetune_metrics, finetune_history = run_finetune_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        base_params,
        n_epochs=n_target_epochs,
    )

    return {
        "base_history": base_history,
        "frozen": frozen_metrics,
        "frozen_history": frozen_history,
        "finetune": finetune_metrics,
        "finetune_history": finetune_history,
        "finetune_params": finetune_params,
        "finetune_head": finetune_head,
    }


__all__ = [
    "run_all_experiments",
    "run_frozen_transfer_experiment",
    "run_finetune_experiment",
]
