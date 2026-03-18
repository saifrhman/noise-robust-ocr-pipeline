#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.receipt_ai.evaluation.field_comparison import COMPARISON_FIELDS, compare_fields


HIGHLIGHT_FIELDS = ["vendor.name", "invoice.date", "totals.total"]
ITEM_ANALYSIS_MODE_KEYS = {
    "easyocr_rules": "easyocr",
    "layoutlm_only": "layoutlm",
    "hybrid": "hybrid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate baseline vs improved experiment summary report.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--improved-checkpoint", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-failures", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_comparison_path = _find_one(experiment_root / "baseline" / "comparison", "comparison_*.json")
    improved_comparison_path = _find_one(experiment_root / "improved" / "comparison", "comparison_*.json")
    baseline_analysis_dir = experiment_root / "baseline" / "analysis"
    improved_analysis_dir = experiment_root / "improved" / "analysis"
    baseline_eval_path = experiment_root / "baseline" / "eval" / "metrics.json"
    improved_eval_path = experiment_root / "improved" / "eval" / "metrics.json"
    training_summary_path = _find_optional(experiment_root / "training", "*/receipt_ai_training_summary.json")
    training_diagnostics_path = _find_optional(experiment_root / "training", "*/receipt_ai_training_diagnostics.json")
    weak_label_summary_path = _find_optional(experiment_root / "training" / "weak_label_analysis", "*_summary.json")
    ablation_path = experiment_root / "ablation" / "policy_ablation_report.json"

    baseline_comparison = _read_json(baseline_comparison_path)
    improved_comparison = _read_json(improved_comparison_path)
    baseline_eval = _read_json(baseline_eval_path)
    improved_eval = _read_json(improved_eval_path)
    training_summary = _read_json(training_summary_path) if training_summary_path else {}
    training_diagnostics = _read_json(training_diagnostics_path) if training_diagnostics_path else {}
    weak_label_summary = _read_json(weak_label_summary_path) if weak_label_summary_path else {}
    ablation = _read_json(ablation_path) if ablation_path.exists() else {}
    baseline_analysis = _load_analysis_dir(baseline_analysis_dir)
    improved_analysis = _load_analysis_dir(improved_analysis_dir)

    baseline_rows = _index_by_sample_id(baseline_comparison)
    improved_rows = _index_by_sample_id(improved_comparison)
    common_sample_ids = sorted(set(baseline_rows) & set(improved_rows))

    model_field_metrics = _mode_field_metrics(common_sample_ids, baseline_rows, improved_rows, mode_name="layoutlm_only")
    hybrid_field_metrics = _mode_field_metrics(common_sample_ids, baseline_rows, improved_rows, mode_name="hybrid")
    model_contribution = _hybrid_provenance_metrics(common_sample_ids, baseline_rows, improved_rows)
    item_coherence = _item_coherence_summary(baseline_analysis.get("item_analyses"), improved_analysis.get("item_analyses"))
    failure_cases = _extract_failure_cases(
        common_sample_ids,
        baseline_rows,
        improved_rows,
        baseline_analysis.get("item_analyses"),
        improved_analysis.get("item_analyses"),
        limit=args.top_failures,
    )

    summary = {
        "experiment_root": str(experiment_root),
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).expanduser().resolve()),
        "improved_checkpoint": str(Path(args.improved_checkpoint).expanduser().resolve()),
        "sample_overlap": {
            "baseline_samples": len(baseline_rows),
            "improved_samples": len(improved_rows),
            "common_samples": len(common_sample_ids),
        },
        "training": {
            "summary_path": str(training_summary_path) if training_summary_path else "",
            "diagnostics_path": str(training_diagnostics_path) if training_diagnostics_path else "",
            "weak_label_summary_path": str(weak_label_summary_path) if weak_label_summary_path else "",
            "config": training_summary.get("config", {}),
            "dataset_diagnostics": training_summary.get("dataset_diagnostics", {}),
            "training_diagnostics": training_diagnostics,
            "weak_label_summary": weak_label_summary,
        },
        "evaluation": {
            "baseline": baseline_eval,
            "improved": improved_eval,
            "delta": _evaluation_delta(baseline_eval, improved_eval),
        },
        "comparison": {
            "model_field_metrics": model_field_metrics,
            "hybrid_field_metrics": hybrid_field_metrics,
            "model_top_changes": _top_field_changes(model_field_metrics.get("delta", {})),
            "hybrid_top_changes": _top_field_changes(hybrid_field_metrics.get("delta", {})),
            "item_coherence": item_coherence,
            "ablation": ablation,
        },
        "model_contribution": model_contribution,
        "hybrid_behavior_change": _hybrid_behavior_change(model_contribution),
        "failure_cases": failure_cases,
        "analysis_artifacts": {
            "baseline_comparison": str(baseline_comparison_path),
            "improved_comparison": str(improved_comparison_path),
            "baseline_analysis_dir": str(baseline_analysis_dir),
            "improved_analysis_dir": str(improved_analysis_dir),
            "ablation_report": str(ablation_path) if ablation else "",
        },
    }

    json_path = output_dir / "experiment_summary.json"
    md_path = output_dir / "experiment_summary.md"
    failure_path = output_dir / "failure_cases.json"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(summary), encoding="utf-8")
    failure_path.write_text(json.dumps(failure_cases, indent=2), encoding="utf-8")

    print(f"Saved experiment summary JSON: {json_path}")
    print(f"Saved experiment summary Markdown: {md_path}")
    print(f"Saved failure cases JSON: {failure_path}")


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} under {root}")
    return matches[0]


def _find_optional(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_analysis_dir(analysis_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ["error_summary", "top_disagreements", "item_analyses", "disagreement_summaries"]:
        path = analysis_dir / f"{name}.json"
        out[name] = _read_json(path) if path.exists() else {}
    return out


def _index_by_sample_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_id", "")): row for row in rows if row.get("sample_id")}


def _mode_field_metrics(
    sample_ids: list[str],
    baseline_rows: dict[str, dict[str, Any]],
    improved_rows: dict[str, dict[str, Any]],
    *,
    mode_name: str,
) -> dict[str, Any]:
    field_types = {field_name: field_type for field_name, field_type in COMPARISON_FIELDS}
    baseline_counts = {field_name: {"truth_hits": 0, "truth_total": 0, "coverage_hits": 0, "coverage_total": 0} for field_name in field_types}
    improved_counts = {field_name: {"truth_hits": 0, "truth_total": 0, "coverage_hits": 0, "coverage_total": 0} for field_name in field_types}

    for sample_id in sample_ids:
        baseline_row = baseline_rows[sample_id]
        improved_row = improved_rows[sample_id]
        ground_truth = baseline_row.get("ground_truth") or improved_row.get("ground_truth")
        baseline_result = (((baseline_row.get("modes") or {}).get(mode_name) or {}).get("result")) or {}
        improved_result = (((improved_row.get("modes") or {}).get(mode_name) or {}).get("result")) or {}
        if not baseline_result or not improved_result:
            continue

        baseline_comps = compare_fields(baseline_result, baseline_result, baseline_result, ground_truth)
        improved_comps = compare_fields(improved_result, improved_result, improved_result, ground_truth)
        baseline_map = {comp.field_name: comp for comp in baseline_comps}
        improved_map = {comp.field_name: comp for comp in improved_comps}

        for field_name in field_types:
            baseline_comp = baseline_map[field_name]
            improved_comp = improved_map[field_name]
            baseline_counts[field_name]["coverage_total"] += 1
            improved_counts[field_name]["coverage_total"] += 1
            if baseline_comp.easyocr_present:
                baseline_counts[field_name]["coverage_hits"] += 1
            if improved_comp.easyocr_present:
                improved_counts[field_name]["coverage_hits"] += 1
            if baseline_comp.truth_present:
                baseline_counts[field_name]["truth_total"] += 1
                if baseline_comp.easyocr_vs_truth:
                    baseline_counts[field_name]["truth_hits"] += 1
            if improved_comp.truth_present:
                improved_counts[field_name]["truth_total"] += 1
                if improved_comp.easyocr_vs_truth:
                    improved_counts[field_name]["truth_hits"] += 1

    return {
        "baseline": _finalize_field_stats(baseline_counts),
        "improved": _finalize_field_stats(improved_counts),
        "delta": _field_metric_delta(baseline_counts, improved_counts),
    }


def _finalize_field_stats(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for field_name, row in counts.items():
        out[field_name] = {
            **row,
            "truth_accuracy": round(row["truth_hits"] / row["truth_total"], 4) if row["truth_total"] > 0 else None,
            "coverage_rate": round(row["coverage_hits"] / row["coverage_total"], 4) if row["coverage_total"] > 0 else None,
        }
    return out


def _field_metric_delta(
    baseline_counts: dict[str, dict[str, int]],
    improved_counts: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | None]]:
    delta: dict[str, dict[str, float | None]] = {}
    for field_name in baseline_counts:
        baseline = baseline_counts[field_name]
        improved = improved_counts[field_name]
        baseline_truth = baseline["truth_hits"] / baseline["truth_total"] if baseline["truth_total"] > 0 else None
        improved_truth = improved["truth_hits"] / improved["truth_total"] if improved["truth_total"] > 0 else None
        baseline_cov = baseline["coverage_hits"] / baseline["coverage_total"] if baseline["coverage_total"] > 0 else None
        improved_cov = improved["coverage_hits"] / improved["coverage_total"] if improved["coverage_total"] > 0 else None
        delta[field_name] = {
            "truth_accuracy_delta": round(improved_truth - baseline_truth, 4) if baseline_truth is not None and improved_truth is not None else None,
            "coverage_rate_delta": round(improved_cov - baseline_cov, 4) if baseline_cov is not None and improved_cov is not None else None,
        }
    return delta


def _evaluation_delta(baseline_eval: dict[str, Any], improved_eval: dict[str, Any]) -> dict[str, float | None]:
    baseline_metrics = (baseline_eval.get("metrics") or {}) if isinstance(baseline_eval, dict) else {}
    improved_metrics = (improved_eval.get("metrics") or {}) if isinstance(improved_eval, dict) else {}
    keys = ["f1", "precision", "recall", "token_accuracy", "overall_accuracy"]
    out: dict[str, float | None] = {}
    for key in keys:
        baseline_val = baseline_metrics.get(key)
        improved_val = improved_metrics.get(key)
        if isinstance(baseline_val, (int, float)) and isinstance(improved_val, (int, float)):
            out[key] = round(float(improved_val) - float(baseline_val), 6)
        else:
            out[key] = None
    for entity in ["VENDOR_NAME", "DATE", "TOTAL"]:
        baseline_entity = (((baseline_metrics.get("critical_fields") or {}).get(entity) or {}).get("f1"))
        improved_entity = (((improved_metrics.get("critical_fields") or {}).get(entity) or {}).get("f1"))
        key = f"{entity.lower()}_f1"
        if isinstance(baseline_entity, (int, float)) and isinstance(improved_entity, (int, float)):
            out[key] = round(float(improved_entity) - float(baseline_entity), 6)
        else:
            out[key] = None
    return out


def _hybrid_provenance_metrics(
    sample_ids: list[str],
    baseline_rows: dict[str, dict[str, Any]],
    improved_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fields = [
        "vendor.name",
        "invoice.date",
        "totals.total",
        "invoice.bill_number",
        "invoice.order_number",
        "invoice.table_number",
        "invoice.cashier",
        "items",
    ]
    baseline_counts = {field: {"layoutlm_only": 0, "easyocr_rules": 0, "other": 0, "total": 0} for field in fields}
    improved_counts = {field: {"layoutlm_only": 0, "easyocr_rules": 0, "other": 0, "total": 0} for field in fields}

    for sample_id in sample_ids:
        for counts, rows in [(baseline_counts, baseline_rows), (improved_counts, improved_rows)]:
            hybrid_result = ((((rows[sample_id].get("modes") or {}).get("hybrid") or {}).get("result")) or {})
            provenance = ((hybrid_result.get("metadata") or {}).get("field_provenance") or {}) if isinstance(hybrid_result, dict) else {}
            for field in fields:
                source = str(provenance.get(field, "") or "")
                counts[field]["total"] += 1
                if source in {"layoutlm_only", "easyocr_rules"}:
                    counts[field][source] += 1
                else:
                    counts[field]["other"] += 1

    return {
        "baseline": _finalize_provenance_counts(baseline_counts),
        "improved": _finalize_provenance_counts(improved_counts),
        "delta": _provenance_delta(baseline_counts, improved_counts),
    }


def _finalize_provenance_counts(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for field, row in counts.items():
        total = max(row["total"], 1)
        out[field] = {
            **row,
            "layoutlm_only_rate": round(row["layoutlm_only"] / total, 4),
            "easyocr_rules_rate": round(row["easyocr_rules"] / total, 4),
        }
    return out


def _provenance_delta(
    baseline_counts: dict[str, dict[str, int]],
    improved_counts: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for field in baseline_counts:
        b_total = max(baseline_counts[field]["total"], 1)
        i_total = max(improved_counts[field]["total"], 1)
        b_model_rate = baseline_counts[field]["layoutlm_only"] / b_total
        i_model_rate = improved_counts[field]["layoutlm_only"] / i_total
        b_rules_rate = baseline_counts[field]["easyocr_rules"] / b_total
        i_rules_rate = improved_counts[field]["easyocr_rules"] / i_total
        out[field] = {
            "layoutlm_only_rate_delta": round(i_model_rate - b_model_rate, 4),
            "easyocr_rules_rate_delta": round(i_rules_rate - b_rules_rate, 4),
        }
    return out


def _hybrid_behavior_change(model_contribution: dict[str, Any]) -> dict[str, Any]:
    delta = model_contribution.get("delta", {})
    increased_model_use = {field: row for field, row in delta.items() if row.get("layoutlm_only_rate_delta", 0.0) > 0}
    decreased_model_use = {field: row for field, row in delta.items() if row.get("layoutlm_only_rate_delta", 0.0) < 0}
    return {
        "increased_model_use_fields": increased_model_use,
        "decreased_model_use_fields": decreased_model_use,
        "model_used_more_on_highlights": {
            field: delta.get(field, {}) for field in HIGHLIGHT_FIELDS
        },
    }


def _top_field_changes(delta: dict[str, dict[str, float | None]]) -> dict[str, list[dict[str, float | str | None]]]:
    rows = []
    for field_name, metrics in delta.items():
        rows.append(
            {
                "field_name": field_name,
                "truth_accuracy_delta": metrics.get("truth_accuracy_delta"),
                "coverage_rate_delta": metrics.get("coverage_rate_delta"),
            }
        )
    truth_rows = [row for row in rows if isinstance(row.get("truth_accuracy_delta"), (int, float))]
    truth_rows.sort(key=lambda row: float(row.get("truth_accuracy_delta", 0.0)), reverse=True)
    return {
        "largest_gains": truth_rows[:5],
        "largest_losses": list(reversed(truth_rows[-5:])),
    }


def _item_coherence_summary(
    baseline_item_payload: dict[str, Any] | None,
    improved_item_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_analyses = _item_analyses_by_sample_id(baseline_item_payload)
    improved_analyses = _item_analyses_by_sample_id(improved_item_payload)
    sample_ids = sorted(set(baseline_analyses) & set(improved_analyses))

    out: dict[str, Any] = {"baseline": {}, "improved": {}, "delta": {}}
    for mode_name in ["layoutlm_only", "hybrid"]:
        base_key = ITEM_ANALYSIS_MODE_KEYS[mode_name]
        baseline_coherent = 0
        improved_coherent = 0
        total = 0
        for sample_id in sample_ids:
            baseline_analysis = baseline_analyses[sample_id]
            improved_analysis = improved_analyses[sample_id]
            total += 1
            if baseline_analysis.get(f"{base_key}_items_coherent", False):
                baseline_coherent += 1
            if improved_analysis.get(f"{base_key}_items_coherent", False):
                improved_coherent += 1
        baseline_rate = baseline_coherent / total if total > 0 else 0.0
        improved_rate = improved_coherent / total if total > 0 else 0.0
        out["baseline"][mode_name] = {"coherent": baseline_coherent, "total": total, "rate": round(baseline_rate, 4)}
        out["improved"][mode_name] = {"coherent": improved_coherent, "total": total, "rate": round(improved_rate, 4)}
        out["delta"][mode_name] = {"rate_delta": round(improved_rate - baseline_rate, 4)}
    return out


def _item_analyses_by_sample_id(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    analyses = payload.get("analyses", []) or []
    return {str(row.get("sample_id", "")): row for row in analyses if row.get("sample_id")}


def _extract_failure_cases(
    sample_ids: list[str],
    baseline_rows: dict[str, dict[str, Any]],
    improved_rows: dict[str, dict[str, Any]],
    baseline_item_payload: dict[str, Any] | None,
    improved_item_payload: dict[str, Any] | None,
    *,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    baseline_item = _item_analyses_by_sample_id(baseline_item_payload)
    improved_item = _item_analyses_by_sample_id(improved_item_payload)
    model_failures: list[dict[str, Any]] = []
    hybrid_failures: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        baseline_row = baseline_rows[sample_id]
        improved_row = improved_rows[sample_id]
        image_path = baseline_row.get("image_path") or improved_row.get("image_path")
        ground_truth = baseline_row.get("ground_truth") or improved_row.get("ground_truth")

        baseline_model = ((((baseline_row.get("modes") or {}).get("layoutlm_only") or {}).get("result")) or {})
        improved_model = ((((improved_row.get("modes") or {}).get("layoutlm_only") or {}).get("result")) or {})
        baseline_hybrid = ((((baseline_row.get("modes") or {}).get("hybrid") or {}).get("result")) or {})
        improved_hybrid = ((((improved_row.get("modes") or {}).get("hybrid") or {}).get("result")) or {})
        if not baseline_model or not improved_model or not baseline_hybrid or not improved_hybrid:
            continue

        baseline_model_comps = {c.field_name: c for c in compare_fields(baseline_model, baseline_model, baseline_model, ground_truth)}
        improved_model_comps = {c.field_name: c for c in compare_fields(improved_model, improved_model, improved_model, ground_truth)}
        baseline_hybrid_comps = {c.field_name: c for c in compare_fields(baseline_hybrid, baseline_hybrid, baseline_hybrid, ground_truth)}
        improved_hybrid_comps = {c.field_name: c for c in compare_fields(improved_hybrid, improved_hybrid, improved_hybrid, ground_truth)}

        model_regressed_fields: list[str] = []
        hybrid_regressed_fields: list[str] = []
        for field_name in HIGHLIGHT_FIELDS:
            base_comp = baseline_model_comps[field_name]
            imp_comp = improved_model_comps[field_name]
            if base_comp.truth_present and imp_comp.truth_present and base_comp.easyocr_vs_truth and not imp_comp.easyocr_vs_truth:
                model_regressed_fields.append(field_name)

            base_hybrid_comp = baseline_hybrid_comps[field_name]
            imp_hybrid_comp = improved_hybrid_comps[field_name]
            if (
                base_hybrid_comp.truth_present
                and imp_hybrid_comp.truth_present
                and base_hybrid_comp.easyocr_vs_truth
                and not imp_hybrid_comp.easyocr_vs_truth
            ):
                hybrid_regressed_fields.append(field_name)

        base_item = baseline_item.get(sample_id, {})
        imp_item = improved_item.get(sample_id, {})
        model_item_regressed = bool(base_item.get("layoutlm_items_coherent", False) and not imp_item.get("layoutlm_items_coherent", False))
        hybrid_item_regressed = bool(base_item.get("hybrid_items_coherent", False) and not imp_item.get("hybrid_items_coherent", False))

        if model_regressed_fields or model_item_regressed:
            model_failures.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "regressed_fields": model_regressed_fields,
                    "item_coherence_regressed": model_item_regressed,
                    "severity_score": (10 * len(model_regressed_fields)) + (2 if model_item_regressed else 0),
                }
            )
        if hybrid_regressed_fields or hybrid_item_regressed:
            hybrid_failures.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "regressed_fields": hybrid_regressed_fields,
                    "item_coherence_regressed": hybrid_item_regressed,
                    "severity_score": (10 * len(hybrid_regressed_fields)) + (2 if hybrid_item_regressed else 0),
                }
            )

    model_failures.sort(key=lambda row: row["severity_score"], reverse=True)
    hybrid_failures.sort(key=lambda row: row["severity_score"], reverse=True)
    return {
        "model_regressions": model_failures[:limit],
        "hybrid_regressions": hybrid_failures[:limit],
    }


def _to_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Experiment Summary\n\n")
    lines.append(f"- Baseline checkpoint: `{summary.get('baseline_checkpoint')}`\n")
    lines.append(f"- Improved checkpoint: `{summary.get('improved_checkpoint')}`\n")
    overlap = summary.get("sample_overlap", {})
    lines.append(f"- Common samples: {overlap.get('common_samples')}\n\n")

    lines.append("## Training Configuration\n\n")
    train = summary.get("training", {})
    config = train.get("config", {}) or {}
    for key in ["model_name", "epochs", "learning_rate", "train_batch_size", "eval_batch_size", "loss_type", "focal_gamma", "drop_noisy_samples", "critical_label_boost", "weak_label_floor"]:
        if key in config:
            lines.append(f"- `{key}`: {config.get(key)}\n")
    dataset_diagnostics = ((train.get("dataset_diagnostics") or {}).get("train") or {})
    if dataset_diagnostics:
        lines.append(f"- Train samples kept: {dataset_diagnostics.get('samples_kept')}\n")
        lines.append(f"- Train samples dropped: {dataset_diagnostics.get('samples_dropped')}\n")
        lines.append(f"- Avg non-O weight: {((dataset_diagnostics.get('label_confidence') or {}).get('avg_non_o_weight'))}\n")
    weak_label_summary = train.get("weak_label_summary", {}) or {}
    if weak_label_summary:
        lines.append(f"- Weak-label drop candidates: {weak_label_summary.get('drop_candidate_samples')}\n")
        lines.append(f"- Weak-label avg non-O confidence: {weak_label_summary.get('avg_non_o_confidence')}\n")

    lines.append("\n## Evaluation Delta\n\n")
    eval_delta = (summary.get("evaluation") or {}).get("delta", {}) or {}
    for key, value in eval_delta.items():
        lines.append(f"- `{key}`: {value}\n")

    lines.append("\n## Highlight Fields\n\n")
    model_delta = (((summary.get("comparison") or {}).get("model_field_metrics") or {}).get("delta") or {})
    hybrid_delta = (((summary.get("comparison") or {}).get("hybrid_field_metrics") or {}).get("delta") or {})
    for field_name in HIGHLIGHT_FIELDS:
        lines.append(
            f"- `{field_name}`: model truth delta={((model_delta.get(field_name) or {}).get('truth_accuracy_delta'))}, "
            f"hybrid truth delta={((hybrid_delta.get(field_name) or {}).get('truth_accuracy_delta'))}\n"
        )

    lines.append("\n## Largest Field Changes\n\n")
    comparison = summary.get("comparison", {}) or {}
    for title, key in [("Model Gains", "model_top_changes"), ("Hybrid Gains", "hybrid_top_changes")]:
        lines.append(f"### {title}\n\n")
        changes = ((comparison.get(key) or {}).get("largest_gains") or [])
        for row in changes[:5]:
            lines.append(f"- `{row.get('field_name')}` truth delta={row.get('truth_accuracy_delta')}\n")
        lines.append("\n")
        losses = ((comparison.get(key) or {}).get("largest_losses") or [])
        if losses:
            lines.append(f"### {title.replace('Gains', 'Losses')}\n\n")
            for row in losses[:5]:
                lines.append(f"- `{row.get('field_name')}` truth delta={row.get('truth_accuracy_delta')}\n")
            lines.append("\n")

    lines.append("\n## Hybrid Model Contribution\n\n")
    behavior = (summary.get("hybrid_behavior_change") or {}).get("model_used_more_on_highlights", {}) or {}
    for field_name in HIGHLIGHT_FIELDS:
        row = behavior.get(field_name, {})
        lines.append(
            f"- `{field_name}`: model-use delta={row.get('layoutlm_only_rate_delta')}, "
            f"rules-use delta={row.get('easyocr_rules_rate_delta')}\n"
        )

    lines.append("\n## Item Coherence\n\n")
    item_delta = (((summary.get("comparison") or {}).get("item_coherence") or {}).get("delta") or {})
    for mode_name in ["layoutlm_only", "hybrid"]:
        lines.append(f"- `{mode_name}` coherence delta: {((item_delta.get(mode_name) or {}).get('rate_delta'))}\n")

    lines.append("\n## Failures To Inspect\n\n")
    failures = summary.get("failure_cases", {}) or {}
    model_failures = failures.get("model_regressions", []) or []
    hybrid_failures = failures.get("hybrid_regressions", []) or []
    if model_failures:
        lines.append("### Model Regressions\n\n")
        for row in model_failures[:10]:
            lines.append(f"- `{row.get('sample_id')}` fields={row.get('regressed_fields')} item_regressed={row.get('item_coherence_regressed')}\n")
    if hybrid_failures:
        lines.append("\n### Hybrid Regressions\n\n")
        for row in hybrid_failures[:10]:
            lines.append(f"- `{row.get('sample_id')}` fields={row.get('regressed_fields')} item_regressed={row.get('item_coherence_regressed')}\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
