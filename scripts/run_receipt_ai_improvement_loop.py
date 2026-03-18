#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the weak-label improvement loop: analyze -> train -> evaluate -> compare."
    )
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    parser.add_argument("--output-root", default="outputs/receipt_ai_improvement_loop")
    parser.add_argument("--experiment-name", default="short_retrain")
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--eval-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_output_dir = output_root / "training"
    checkpoint_dir = train_output_dir / args.experiment_name
    weak_label_dir = output_root / "weak_label_analysis"
    eval_dir = output_root / "eval"
    comparison_dir = output_root / "comparison"

    commands = [
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
        [
            sys.executable,
            "scripts/train_receipt_ai_layoutlmv3.py",
            "--dataset-root",
            args.dataset_root,
            "--model-name",
            args.base_model,
            "--output-dir",
            str(train_output_dir),
            "--experiment-name",
            args.experiment_name,
            "--train-split",
            args.train_split,
            "--val-split",
            args.eval_split,
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
            str(args.max-eval-samples),
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
        [
            sys.executable,
            "scripts/evaluate_receipt_ai_layoutlmv3.py",
            "--dataset-root",
            args.dataset_root,
            "--split",
            args.eval_split,
            "--checkpoint",
            str(checkpoint_dir),
            "--max-samples",
            str(args.max-eval-samples),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(eval_dir),
        ],
        [
            sys.executable,
            "scripts/compare_extraction_modes_receipt_ai.py",
            "--dataset-root",
            args.dataset_root,
            "--checkpoint",
            str(checkpoint_dir),
            "--split",
            args.eval_split,
            "--max-samples",
            str(args.max-eval-samples),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(comparison_dir),
        ],
    ]

    for command in commands:
        print(f"Running: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)

    print("\nImprovement loop complete")
    print(f"Weak-label analysis: {weak_label_dir}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Evaluation: {eval_dir}")
    print(f"Mode comparison: {comparison_dir}")


if __name__ == "__main__":
    main()
