#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.model.weak_label_analysis import analyze_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak-label quality summary and optional baseline delta.")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--item-name-threshold", type=int, default=30)
    parser.add_argument("--baseline-summary", default="", help="Optional previous weak-label summary JSON to compare")
    parser.add_argument("--output-dir", default="outputs/weak_label_quality_delta")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()

    loader = SROIEDatasetLoader(cfg.paths.data_root)
    summary, suspicious = analyze_split(
        loader,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict=False,
        max_samples=args.max_samples,
        item_name_threshold=args.item_name_threshold,
    )

    baseline = None
    if args.baseline_summary:
        baseline_path = Path(args.baseline_summary).expanduser().resolve()
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    delta = _compute_delta(baseline, summary)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_path = output_dir / f"{args.split}_current_summary.json"
    suspicious_path = output_dir / f"{args.split}_suspicious_samples.json"
    delta_path = output_dir / f"{args.split}_delta.json"

    current_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    suspicious_path.write_text(json.dumps(suspicious, indent=2), encoding="utf-8")
    delta_path.write_text(json.dumps(delta, indent=2), encoding="utf-8")

    print(f"Saved current summary: {current_path}")
    print(f"Saved suspicious samples: {suspicious_path}")
    print(f"Saved delta summary: {delta_path}")


def _compute_delta(baseline: dict | None, current: dict) -> dict:
    out = {
        "has_baseline": baseline is not None,
        "suspicious_count_current": int(current.get("suspicious_sample_count", 0)),
    }
    if baseline is None:
        out["note"] = "No baseline provided; delta only includes current summary signals."
        return out

    base_susp = int(baseline.get("suspicious_sample_count", 0))
    cur_susp = int(current.get("suspicious_sample_count", 0))
    out["suspicious_count_baseline"] = base_susp
    out["suspicious_count_delta"] = cur_susp - base_susp

    base_item = baseline.get("item_summary", {}) if isinstance(baseline, dict) else {}
    cur_item = current.get("item_summary", {}) if isinstance(current, dict) else {}
    for key in ["items_detected", "items_labeled", "items_missing_price", "name_spans", "price_spans"]:
        out[f"item_{key}_baseline"] = int(base_item.get(key, 0))
        out[f"item_{key}_current"] = int(cur_item.get(key, 0))
        out[f"item_{key}_delta"] = int(cur_item.get(key, 0)) - int(base_item.get(key, 0))

    return out


if __name__ == "__main__":
    main()
