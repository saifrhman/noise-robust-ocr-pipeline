#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnosis, recommendation generation, failure bucketing, and promotion decision.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_root / "diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            sys.executable,
            "scripts/diagnose_experiment_results_receipt_ai.py",
            "--experiment-root",
            str(experiment_root),
            "--output-dir",
            str(output_dir),
        ],
        [
            sys.executable,
            "scripts/generate_next_step_recommendations_receipt_ai.py",
            "--experiment-root",
            str(experiment_root),
            "--output-dir",
            str(output_dir),
        ],
        [
            sys.executable,
            "scripts/bucket_failure_cases_receipt_ai.py",
            "--experiment-root",
            str(experiment_root),
            "--output-dir",
            str(output_dir),
        ],
        [
            sys.executable,
            "scripts/decide_checkpoint_promotion_receipt_ai.py",
            "--experiment-root",
            str(experiment_root),
            "--output-dir",
            str(output_dir),
        ],
    ]

    for command in commands:
        print(f"Running: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)

    print("\nExperiment postmortem complete")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
