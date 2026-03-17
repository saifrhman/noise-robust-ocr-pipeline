#!/usr/bin/env python3
"""
Compare easyocr_rules, layoutlm_only, and hybrid extraction modes on the same samples.

Runs all 3 modes on a validation/test subset and collects per-sample outputs for
comparison analysis and error investigation.

Usage:
    python scripts/compare_extraction_modes_receipt_ai.py --split val --max-samples 50 --seed 42
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.pipelines.entrypoints import (
    run_easyocr_rules,
    run_layoutlm_only,
    run_hybrid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare extraction modes on same sample set."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to process (0=all)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio for train split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/comparison",
        help="Output directory for comparison artifacts",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/SROIE2019",
        help="Root directory for SROIE dataset",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first error instead of collecting all",
    )
    return parser.parse_args()


def run_all_modes(
    image_path: str | Path,
    config: ReceiptAIConfig,
) -> dict[str, Any]:
    """Run all 3 modes on same image, collect results and metadata."""
    results = {}
    
    # Run each mode
    for mode_name, runner in [
        ("easyocr_rules", run_easyocr_rules),
        ("layoutlm_only", run_layoutlm_only),
        ("hybrid", run_hybrid),
    ]:
        try:
            result = runner(image_path, config=config)
            results[mode_name] = {
                "status": "success",
                "result": result.to_dict(),
                "error": None,
            }
        except Exception as e:
            results[mode_name] = {
                "status": "error",
                "result": None,
                "error": str(e),
            }
    
    return results


def main() -> None:
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()
    
    # Prepare output dir
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Comparing extraction modes on {args.split} split")
    print(f"Output directory: {output_dir}")
    
    loader = SROIEDatasetLoader(cfg.paths.data_root)
    
    # Collect per-sample results
    sample_results = []
    processed = 0
    failed = 0
    
    for sample in loader.iter_samples(
        args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict=False,
    ):
        if args.max_samples > 0 and processed >= args.max_samples:
            break
        
        try:
            # Run all modes
            mode_outputs = run_all_modes(sample.image_path, cfg)
            
            # Collect per-sample data
            sample_data = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "modes": mode_outputs,
                "ground_truth": sample.ground_truth.to_dict() if sample.ground_truth else None,
            }
            sample_results.append(sample_data)
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed} samples...", flush=True)
        
        except Exception as e:
            if args.strict:
                raise
            failed += 1
            print(f"  ⚠ Failed to process {sample.sample_id}: {e}", flush=True)
    
    # Save all per-sample results
    output_file = output_dir / f"comparison_{args.split}.json"
    with open(output_file, "w") as f:
        json.dump(sample_results, f, indent=2)
    
    print(f"\n✓ Processed {processed} samples, {failed} errors")
    print(f"✓ Saved per-sample results to {output_file}")


if __name__ == "__main__":
    main()
