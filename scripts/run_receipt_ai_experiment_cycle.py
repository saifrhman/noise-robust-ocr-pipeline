#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full baseline vs improved receipt-ai experiment cycle.")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-dir", default="outputs/experiments")
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.output_dir).expanduser().resolve() / args.experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = Path(args.baseline_checkpoint).expanduser().resolve()
    if not baseline_checkpoint.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")

    weak_label_dir = experiment_root / "training" / "weak_label_analysis"
    training_output_dir = experiment_root / "training" / "checkpoints"
    improved_checkpoint = training_output_dir / args.experiment_name

    commands: list[tuple[list[str], dict[str, str] | None]] = [
        (
            [
                sys.executable,
                "scripts/analyze_weak_labels_receipt_ai.py",
                "--dataset-root",
                args.dataset_root,
                "--split",
                args.train_split,
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--max-samples",
                str(args.max_train_samples),
                "--output-dir",
                str(weak_label_dir),
            ],
            None,
        ),
        (
            [
                sys.executable,
                "scripts/train_receipt_ai_layoutlmv3.py",
                "--dataset-root",
                args.dataset_root,
                "--model-name",
                args.base_model,
                "--output-dir",
                str(training_output_dir),
                "--experiment-name",
                args.experiment_name,
                "--train-split",
                args.train_split,
                "--val-split",
                args.split,
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--train-batch-size",
                str(args.train_batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--max-train-samples",
                str(args.max_train_samples),
                "--max-val-samples",
                str(args.max_eval_samples),
                "--loss-type",
                "focal",
                "--focal-gamma",
                "2.0",
                "--use-class-weights",
                "--oversample-non-o",
                "--drop-noisy-samples",
                "--critical-label-boost",
                "1.75",
                "--weak-label-floor",
                "0.40",
            ],
            None,
        ),
        (
            [
                sys.executable,
                "scripts/evaluate_receipt_ai_layoutlmv3.py",
                "--dataset-root",
                args.dataset_root,
                "--split",
                args.split,
                "--checkpoint",
                str(baseline_checkpoint),
                "--max-samples",
                str(args.max_eval_samples),
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(experiment_root / "baseline" / "eval"),
            ],
            None,
        ),
        (
            [
                sys.executable,
                "scripts/evaluate_receipt_ai_layoutlmv3.py",
                "--dataset-root",
                args.dataset_root,
                "--split",
                args.split,
                "--checkpoint",
                str(improved_checkpoint),
                "--max-samples",
                str(args.max_eval_samples),
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(experiment_root / "improved" / "eval"),
            ],
            None,
        ),
        (
            [
                sys.executable,
                "scripts/compare_extraction_modes_receipt_ai.py",
                "--dataset-root",
                args.dataset_root,
                "--checkpoint",
                str(baseline_checkpoint),
                "--split",
                args.split,
                "--max-samples",
                str(args.max_eval_samples),
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(experiment_root / "baseline" / "comparison"),
            ],
            None,
        ),
        (
            [
                sys.executable,
                "scripts/compare_extraction_modes_receipt_ai.py",
                "--dataset-root",
                args.dataset_root,
                "--checkpoint",
                str(improved_checkpoint),
                "--split",
                args.split,
                "--max-samples",
                str(args.max_eval_samples),
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(experiment_root / "improved" / "comparison"),
            ],
            None,
        ),
        (
            [
                sys.executable,
                "scripts/analyze_comparison_receipt_ai.py",
                "--comparison-file",
                str(_comparison_path(experiment_root / "baseline" / "comparison", args.split)),
                "--output-dir",
                str(experiment_root / "baseline" / "analysis"),
            ],
            None,
        ),
        (
            [
                sys.executable,
                "scripts/analyze_comparison_receipt_ai.py",
                "--comparison-file",
                str(_comparison_path(experiment_root / "improved" / "comparison", args.split)),
                "--output-dir",
                str(experiment_root / "improved" / "analysis"),
            ],
            None,
        ),
        (
            [
                sys.executable,
                "scripts/evaluate_policy_ablation_receipt_ai.py",
                "--baseline-comparison-file",
                str(_comparison_path(experiment_root / "baseline" / "comparison", args.split)),
                "--output-dir",
                str(experiment_root / "ablation"),
                "--max-samples",
                str(args.max_eval_samples),
            ],
            {"RECEIPT_MODEL_CHECKPOINT": str(improved_checkpoint)},
        ),
        (
            [
                sys.executable,
                "scripts/generate_experiment_report_receipt_ai.py",
                "--experiment-root",
                str(experiment_root),
                "--baseline-checkpoint",
                str(baseline_checkpoint),
                "--improved-checkpoint",
                str(improved_checkpoint),
                "--output-dir",
                str(experiment_root / "report"),
            ],
            None,
        ),
    ]

    for command, env_overrides in commands:
        print(f"Running: {' '.join(command)}", flush=True)
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        subprocess.run(command, check=True, env=env)

    print("\nExperiment cycle complete")
    print(f"Experiment root: {experiment_root}")
    print(f"Improved checkpoint: {improved_checkpoint}")
    print(f"Report: {experiment_root / 'report' / 'experiment_summary.md'}")


def _comparison_path(directory: Path, split: str) -> Path:
    return directory / f"comparison_{split}.json"


if __name__ == "__main__":
    main()
