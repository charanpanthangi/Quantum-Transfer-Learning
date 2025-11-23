"""SVG plotting utilities for Quantum Transfer Learning results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import numpy as np

# Use seaborn style for readability
sns.set(style="whitegrid")


def _prepare_svg():
    """Ensure matplotlib uses SVG output."""

    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["figure.dpi"] = 120


def plot_feature_space_source_vs_target(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    output_path: str = "examples/qtl_feature_space_source_vs_target.svg",
):
    """Plot PCA projection of source vs target feature spaces and save SVG."""

    _prepare_svg()
    pca = PCA(n_components=2)
    combined = np.vstack([source_features, target_features])
    proj = pca.fit_transform(combined)
    src_proj = proj[: len(source_features)]
    tgt_proj = proj[len(source_features) :]

    plt.figure(figsize=(6, 4))
    plt.scatter(src_proj[:, 0], src_proj[:, 1], c=source_labels, cmap="Blues", label="Source")
    plt.scatter(tgt_proj[:, 0], tgt_proj[:, 1], c=target_labels, cmap="Oranges", label="Target", marker="x")
    plt.title("PQC feature space: source vs target")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_training_curves_frozen_vs_finetune(
    history_frozen: list,
    history_finetune: list,
    output_path: str = "examples/qtl_training_curves_frozen_vs_finetune.svg",
):
    """Plot loss curves for frozen head training vs fine-tuning."""

    _prepare_svg()
    plt.figure(figsize=(6, 4))
    frozen_losses = [h["loss"] for h in history_frozen]
    finetune_losses = [h["train_loss"] for h in history_finetune]
    plt.plot(frozen_losses, label="Frozen PQC (head only)")
    plt.plot(finetune_losses, label="Fine-tune PQC + head")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Target-task training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_accuracy_comparison(
    acc_frozen: float,
    acc_finetune: float,
    output_path: str = "examples/qtl_accuracy_comparison.svg",
):
    """Bar chart comparing target accuracies."""

    _prepare_svg()
    plt.figure(figsize=(5, 4))
    methods = ["Frozen PQC", "Fine-tune"]
    accuracies = [acc_frozen, acc_finetune]
    sns.barplot(x=methods, y=accuracies, palette=["#4c72b0", "#dd8452"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Target-task accuracy comparison")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


__all__ = [
    "plot_feature_space_source_vs_target",
    "plot_training_curves_frozen_vs_finetune",
    "plot_accuracy_comparison",
]
