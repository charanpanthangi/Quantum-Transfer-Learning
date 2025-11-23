"""CLI entrypoint for running Quantum Transfer Learning experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import dataset
from .plots import (
    plot_accuracy_comparison,
    plot_feature_space_source_vs_target,
    plot_training_curves_frozen_vs_finetune,
)
from .transfer_pipeline import run_all_experiments
from .feature_extractor import extract_features


def parse_args():
    """Parse command-line arguments with sensible defaults."""

    parser = argparse.ArgumentParser(description="Quantum Transfer Learning demo")
    parser.add_argument("--source-samples", type=int, default=200, help="Number of source samples")
    parser.add_argument("--target-samples", type=int, default=200, help="Number of target samples")
    parser.add_argument("--epochs-base", type=int, default=20, help="Epochs for source pretraining")
    parser.add_argument("--epochs-target", type=int, default=25, help="Epochs for target training")
    parser.add_argument("--output-dir", type=str, default="examples", help="Where to save SVG plots")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate datasets
    X_source, y_source = dataset.generate_source_dataset(n_samples=args.source_samples)
    X_target, y_target = dataset.generate_target_dataset(n_samples=args.target_samples)

    # 2. Run experiments
    results = run_all_experiments(
        X_source,
        y_source,
        X_target,
        y_target,
        n_base_epochs=args.epochs_base,
        n_target_epochs=args.epochs_target,
    )

    # 3. Extract features for visualization
    base_params = results["finetune_params"]  # after fine-tuning for better separation
    source_features = extract_features(X_source, base_params)
    target_features = extract_features(X_target, base_params)

    # 4. Plot SVG figures
    plot_feature_space_source_vs_target(
        source_features,
        y_source,
        target_features,
        y_target,
        output_path=str(output_dir / "qtl_feature_space_source_vs_target.svg"),
    )
    plot_training_curves_frozen_vs_finetune(
        results["frozen_history"]["head"],
        results["finetune_history"]["joint"],
        output_path=str(output_dir / "qtl_training_curves_frozen_vs_finetune.svg"),
    )
    plot_accuracy_comparison(
        results["frozen"]["accuracy"],
        results["finetune"]["accuracy"],
        output_path=str(output_dir / "qtl_accuracy_comparison.svg"),
    )

    print("Frozen PQC + head target accuracy:", results["frozen"]["accuracy"])
    print("Fine-tuned PQC + head target accuracy:", results["finetune"]["accuracy"])
    print(f"SVG plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
