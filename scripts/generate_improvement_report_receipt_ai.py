#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.receipt_ai.evaluation.improvement_report import ErrorDrivenImprovementReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate error-driven improvement report from analysis artifacts.")
    parser.add_argument("--comparison-file", required=True, help="Comparison JSON produced by compare_extraction_modes")
    parser.add_argument("--analysis-dir", required=True, help="Analysis directory containing error_summary/top_disagreements/item_analyses")
    parser.add_argument("--output-dir", default="outputs/improvement", help="Directory to save report artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_file = Path(args.comparison_file).expanduser().resolve()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reporter = ErrorDrivenImprovementReporter(comparison_file, analysis_dir)
    report = reporter.generate()

    json_path = output_dir / "error_driven_improvement_report.json"
    md_path = output_dir / "error_driven_improvement_report.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md_path.write_text(reporter.to_markdown(report), encoding="utf-8")

    print(f"Saved machine-readable report: {json_path}")
    print(f"Saved markdown summary: {md_path}")


if __name__ == "__main__":
    main()
