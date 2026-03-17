#!/usr/bin/env python3
"""
Comprehensive comparison and error analysis script.

Orchestrates mode comparison, field metrics, disagreement analysis,
error bucketing, and generates human-review artifacts.

Usage:
    # First run mode comparison to generate comparison_val.json:
    python scripts/compare_extraction_modes_receipt_ai.py --split val --max-samples 50

    # Then run this analysis:
    python scripts/analyze_comparison_receipt_ai.py --comparison-file outputs/comparison/comparison_val.json
"""

import sys
import json
import argparse
import csv
from pathlib import Path
from typing import Any
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.receipt_ai.evaluation import (
    compare_fields,
    analyze_items,
    analyze_disagreements,
    summarize_sample_disagreements,
    bucket_errors,
    summarize_all_errors,
    format_error_report_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze mode comparison results and generate error reports."
    )
    parser.add_argument(
        "--comparison-file",
        required=True,
        help="Path to comparison_*.json from compare_extraction_modes script",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis",
        help="Output directory for analysis artifacts",
    )
    parser.add_argument(
        "--critical-fields-only",
        action="store_true",
        help="Focus on vendor/date/total only",
    )
    parser.add_argument(
        "--top-disagreements",
        type=int,
        default=10,
        help="Number of top disagreement cases to save",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load comparison results
    comparison_file = Path(args.comparison_file).expanduser().resolve()
    if not comparison_file.exists():
        print(f"Error: Comparison file not found: {comparison_file}")
        sys.exit(1)
    
    with open(comparison_file) as f:
        sample_results = json.load(f)
    
    print(f"Loaded {len(sample_results)} samples from {comparison_file.name}")
    
    # Prepare output directory
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    all_field_comparisons = []
    all_disagreement_summaries = []
    all_error_buckets = []
    all_item_analyses = []
    top_disagreements_list = []
    
    for sample_data in sample_results:
        sample_id = sample_data.get("sample_id", "unknown")
        
        # Extract results from each mode
        modes = sample_data.get("modes", {})
        
        easyocr_result = modes.get("easyocr_rules", {}).get("result", {})
        layoutlm_result = modes.get("layoutlm_only", {}).get("result", {})
        hybrid_result = modes.get("hybrid", {}).get("result", {})
        ground_truth = sample_data.get("ground_truth")
        
        # Check for run errors
        if not easyocr_result or not layoutlm_result or not hybrid_result:
            print(f"  ⚠ {sample_id}: One or more modes failed, skipping detailed analysis")
            continue
        
        # Field comparison
        field_comps = compare_fields(
            easyocr_result,
            layoutlm_result,
            hybrid_result,
            ground_truth,
        )
        all_field_comparisons.extend(field_comps)
        
        # Item analysis
        item_analysis = analyze_items(
            easyocr_result,
            layoutlm_result,
            hybrid_result,
            sample_id=sample_id,
        )
        all_item_analyses.append(item_analysis)
        
        # Disagreement analysis
        disagreements = analyze_disagreements(
            field_comps,
            sample_id=sample_id,
            ground_truth_available=(ground_truth is not None),
        )
        
        disagree_summary = summarize_sample_disagreements(
            disagreements,
            field_comps,
            sample_id=sample_id,
        )
        all_disagreement_summaries.append(disagree_summary)
        
        # Collect top disagreements for manual review
        for case in disagreements[:args.top_disagreements]:
            top_disagreements_list.append(case)
        
        # Error bucketing
        error_buckets = bucket_errors(
            sample_id,
            field_comps,
            easyocr_result,
            layoutlm_result,
            hybrid_result,
        )
        all_error_buckets.append(error_buckets)
    
    print(f"Analyzed {len(all_field_comparisons)} field comparisons")
    print(f"Found {len(top_disagreements_list)} top disagreement cases")
    print(f"Identified {len(all_error_buckets)} samples with potential errors")
    
    # Save artifacts
    print("\nGenerating artifacts...")
    
    # 1. Per-sample field comparison (JSON)
    field_comp_file = output_dir / "field_comparisons.json"
    field_comp_data = {
        "total_fields_compared": len(all_field_comparisons),
        "comparisons": [c.to_dict() for c in all_field_comparisons],
    }
    with open(field_comp_file, "w") as f:
        json.dump(field_comp_data, f, indent=2)
    print(f"✓ Field comparisons: {field_comp_file}")
    
    # 2. Per-sample disagreement summaries (JSON)
    disagree_file = output_dir / "disagreement_summaries.json"
    disagree_data = {
        "total_samples": len(all_disagreement_summaries),
        "summaries": [s.to_dict() for s in all_disagreement_summaries],
    }
    with open(disagree_file, "w") as f:
        json.dump(disagree_data, f, indent=2)
    print(f"✓ Disagreement summaries: {disagree_file}")
    
    # 3. Top disagreement cases (JSON + CSV for easy review)
    top_disagree_json = output_dir / "top_disagreements.json"
    with open(top_disagree_json, "w") as f:
        json.dump(
            [c.to_dict() for c in top_disagreements_list],
            f,
            indent=2,
        )
    print(f"✓ Top disagreements (JSON): {top_disagree_json}")
    
    # Also save as CSV for spreadsheet review
    top_disagree_csv = output_dir / "top_disagreements.csv"
    if top_disagreements_list:
        with open(top_disagree_csv, "w", newline="") as f:
            fieldnames = [
                "sample_id",
                "field_name",
                "disagreement_type",
                "severity",
                "easyocr_value",
                "layoutlm_value",
                "hybrid_value",
                "explanation",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for case in top_disagreements_list:
                writer.writerow({
                    "sample_id": case.sample_id,
                    "field_name": case.field_name,
                    "disagreement_type": case.disagreement_type,
                    "severity": case.severity,
                    "easyocr_value": case.easyocr_value,
                    "layoutlm_value": case.layoutlm_value,
                    "hybrid_value": case.hybrid_value,
                    "explanation": case.explanation,
                })
        print(f"✓ Top disagreements (CSV): {top_disagree_csv}")
    
    # 4. Item analysis (JSON)
    item_analysis_file = output_dir / "item_analyses.json"
    item_data = {
        "total_samples": len(all_item_analyses),
        "analyses": [a.to_dict() for a in all_item_analyses],
    }
    with open(item_analysis_file, "w") as f:
        json.dump(item_data, f, indent=2)
    print(f"✓ Item analyses: {item_analysis_file}")
    
    # 5. Error bucketing and summary
    error_file = output_dir / "error_buckets.json"
    error_data = {
        "total_samples": len(all_error_buckets),
        "buckets": [b.to_dict() for b in all_error_buckets],
    }
    with open(error_file, "w") as f:
        json.dump(error_data, f, indent=2)
    print(f"✓ Error buckets: {error_file}")
    
    # Error summary
    error_summary = summarize_all_errors(all_error_buckets)
    error_summary_file = output_dir / "error_summary.json"
    with open(error_summary_file, "w") as f:
        json.dump(error_summary.to_dict(), f, indent=2)
    print(f"✓ Error summary: {error_summary_file}")
    
    # Error markdown report
    error_markdown_file = output_dir / "error_report.md"
    with open(error_markdown_file, "w") as f:
        f.write(format_error_report_markdown(error_summary, all_error_buckets))
    print(f"✓ Error markdown report: {error_markdown_file}")
    
    # 6. Comprehensive sample-by-sample review
    sample_review_file = output_dir / "sample_review.jsonl"
    with open(sample_review_file, "w") as f:
        for sample_data in sample_results:
            sample_id = sample_data.get("sample_id", "unknown")
            
            # Find related analysis data
            disagree_summary = next(
                (s for s in all_disagreement_summaries if s.sample_id == sample_id),
                None,
            )
            item_analysis = next(
                (a for a in all_item_analyses if a.sample_id == sample_id),
                None,
            )
            error_buckets = next(
                (e for e in all_error_buckets if e.sample_id == sample_id),
                None,
            )
            
            review_entry = {
                "sample_id": sample_id,
                "image_path": sample_data.get("image_path"),
                "disagreement_summary": disagree_summary.to_dict() if disagree_summary else None,
                "item_analysis": item_analysis.to_dict() if item_analysis else None,
                "error_buckets": error_buckets.to_dict() if error_buckets else None,
            }
            f.write(json.dumps(review_entry) + "\n")
    print(f"✓ Sample review (JSONL): {sample_review_file}")
    
    # 7. Summary report
    summary_report_file = output_dir / "summary_report.md"
    with open(summary_report_file, "w") as f:
        f.write("# Mode Comparison and Analysis Report\n\n")
        f.write(f"Total samples analyzed: {len(sample_results)}\n")
        f.write(f"Total field comparisons: {len(all_field_comparisons)}\n")
        f.write(f"Samples with disagreements: {sum(1 for s in all_disagreement_summaries if s.total_disagreements > 0)}\n")
        f.write(f"Critical field issues: {sum(s.critical_field_issues for s in all_disagreement_summaries)}\n")
        f.write(f"\n## Error Summary\n\n")
        f.write(f"Total errors identified: {error_summary.total_errors}\n")
        f.write(f"Samples with critical errors: {error_summary.critical_error_samples}\n\n")
        f.write(f"## Quick Actions\n\n")
        f.write(f"1. Review top disagreements: `{top_disagree_csv.name}`\n")
        f.write(f"2. Check error buckets: `{error_markdown_file.name}`\n")
        f.write(f"3. Manual sample review: `{sample_review_file.name}`\n")
        f.write(f"4. Item analysis details: `{item_analysis_file.name}`\n")
    print(f"✓ Summary report: {summary_report_file}")
    
    print(f"\n✓ Analysis complete. artifacts in: {output_dir}")
    print("\nKey artifacts:")
    print(f"  - {top_disagree_csv.name} -- review in spreadsheet")
    print(f"  - {error_markdown_file.name} -- human-readable error summary")
    print(f"  - {sample_review_file.name} -- per-sample detailed analysis")


if __name__ == "__main__":
    main()
