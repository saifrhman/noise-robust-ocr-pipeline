#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run larger validation workflow for higher-confidence default-policy evidence.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample-cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--base-output-dir", default="outputs/validation_runs")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("valrun_%Y%m%d_%H%M%S")
    root = Path(args.base_output_dir).expanduser().resolve() / run_name
    comparison_dir = root / "comparison"
    analysis_dir = root / "analysis"
    improvement_dir = root / "improvement"
    ablation_dir = root / "ablation"
    runtime_dir = root / "runtime"

    comparison_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    improvement_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    comparison_file = comparison_dir / f"comparison_{args.split}.json"

    if not (args.reuse_cache and comparison_file.exists()):
        _run(
            [
                "python",
                "scripts/compare_extraction_modes_receipt_ai.py",
                "--dataset-root",
                args.dataset_root,
                "--split",
                args.split,
                "--max-samples",
                str(args.sample_cap),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(comparison_dir),
            ]
        )
    else:
        print(f"Reusing cached comparison file: {comparison_file}")

    _run(
        [
            "python",
            "scripts/analyze_comparison_receipt_ai.py",
            "--comparison-file",
            str(comparison_file),
            "--output-dir",
            str(analysis_dir),
        ]
    )

    _run(
        [
            "python",
            "scripts/generate_improvement_report_receipt_ai.py",
            "--comparison-file",
            str(comparison_file),
            "--analysis-dir",
            str(analysis_dir),
            "--output-dir",
            str(improvement_dir),
        ]
    )

    _run(
        [
            "python",
            "scripts/evaluate_policy_ablation_receipt_ai.py",
            "--baseline-comparison-file",
            str(comparison_file),
            "--max-samples",
            str(args.sample_cap),
            "--output-dir",
            str(ablation_dir),
        ]
    )

    _run(
        [
            "python",
            "scripts/select_best_policy_receipt_ai.py",
            "--ablation-report",
            str(ablation_dir / "policy_ablation_report.json"),
            "--improvement-report",
            str(improvement_dir / "error_driven_improvement_report.json"),
            "--comparison-file",
            str(comparison_file),
            "--sample-size-threshold",
            str(max(50, args.sample_cap // 2)),
            "--output-dir",
            str(runtime_dir),
        ]
    )

    print("Large validation workflow complete")
    print(f"Run directory: {root}")
    print(f"Recommendation: {runtime_dir / 'best_policy_recommendation.json'}")


def _run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
