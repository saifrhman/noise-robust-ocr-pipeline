#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CRITICAL_FIELDS = ["vendor.name", "invoice.date", "totals.total"]
SECONDARY_FIELDS = ["invoice.bill_number", "invoice.order_number", "invoice.table_number"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose receipt-ai experiment results from experiment artifacts.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_root / "diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(experiment_root / "report" / "experiment_summary.json")
    failure_cases = _read_json(experiment_root / "report" / "failure_cases.json")

    model_delta = (((summary.get("comparison") or {}).get("model_field_metrics") or {}).get("delta") or {})
    hybrid_delta = (((summary.get("comparison") or {}).get("hybrid_field_metrics") or {}).get("delta") or {})
    contribution_delta = (((summary.get("model_contribution") or {}).get("delta")) or {})
    eval_delta = (((summary.get("evaluation") or {}).get("delta")) or {})
    item_delta = ((((summary.get("comparison") or {}).get("item_coherence") or {}).get("delta")) or {})
    sample_overlap = (summary.get("sample_overlap") or {})

    diagnosis = {
        "experiment_root": str(experiment_root),
        "sample_overlap": sample_overlap,
        "headline": {
            "improved_model_beats_baseline": _model_beats_baseline(eval_delta, model_delta),
            "hybrid_improved_over_baseline": _hybrid_improved(hybrid_delta),
            "model_contribution_increased_on_critical_fields": _model_contribution_increased(contribution_delta, CRITICAL_FIELDS),
        },
        "critical_field_summary": _field_group_summary(model_delta, hybrid_delta, contribution_delta, CRITICAL_FIELDS),
        "secondary_field_summary": _field_group_summary(model_delta, hybrid_delta, contribution_delta, SECONDARY_FIELDS),
        "item_coherence_summary": item_delta,
        "regression_summary": _regression_summary(failure_cases),
        "bottleneck_diagnosis": _bottleneck_diagnosis(
            sample_overlap=sample_overlap,
            eval_delta=eval_delta,
            model_delta=model_delta,
            hybrid_delta=hybrid_delta,
            contribution_delta=contribution_delta,
            item_delta=item_delta,
            failure_cases=failure_cases,
        ),
    }

    json_path = output_dir / "diagnosis_report.json"
    md_path = output_dir / "diagnosis_report.md"
    json_path.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(diagnosis), encoding="utf-8")

    print(f"Saved diagnosis JSON: {json_path}")
    print(f"Saved diagnosis Markdown: {md_path}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_beats_baseline(eval_delta: dict[str, Any], model_delta: dict[str, Any]) -> bool:
    overall = float(eval_delta.get("f1") or 0.0)
    critical = [
        float(((model_delta.get(field) or {}).get("truth_accuracy_delta")) or 0.0)
        for field in CRITICAL_FIELDS
    ]
    return overall > 0 and sum(1 for value in critical if value > 0) >= 2


def _hybrid_improved(hybrid_delta: dict[str, Any]) -> bool:
    critical = [
        float(((hybrid_delta.get(field) or {}).get("truth_accuracy_delta")) or 0.0)
        for field in CRITICAL_FIELDS
    ]
    return sum(1 for value in critical if value > 0) >= 2


def _model_contribution_increased(contribution_delta: dict[str, Any], fields: list[str]) -> bool:
    return sum(1 for field in fields if float(((contribution_delta.get(field) or {}).get("layoutlm_only_rate_delta")) or 0.0) > 0) >= 2


def _field_group_summary(
    model_delta: dict[str, Any],
    hybrid_delta: dict[str, Any],
    contribution_delta: dict[str, Any],
    fields: list[str],
) -> dict[str, Any]:
    rows = []
    for field in fields:
        rows.append(
            {
                "field": field,
                "model_truth_delta": ((model_delta.get(field) or {}).get("truth_accuracy_delta")),
                "hybrid_truth_delta": ((hybrid_delta.get(field) or {}).get("truth_accuracy_delta")),
                "model_use_delta": ((contribution_delta.get(field) or {}).get("layoutlm_only_rate_delta")),
            }
        )
    return {"fields": rows}


def _regression_summary(failure_cases: dict[str, Any]) -> dict[str, Any]:
    model_rows = failure_cases.get("model_regressions", []) or []
    hybrid_rows = failure_cases.get("hybrid_regressions", []) or []
    field_counts: dict[str, int] = {}
    for row in [*model_rows, *hybrid_rows]:
        for field in row.get("regressed_fields", []) or []:
            field_counts[field] = field_counts.get(field, 0) + 1
    ranked = sorted(field_counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "model_regression_count": len(model_rows),
        "hybrid_regression_count": len(hybrid_rows),
        "top_regressed_fields": [{"field": field, "count": count} for field, count in ranked[:10]],
    }


def _bottleneck_diagnosis(
    *,
    sample_overlap: dict[str, Any],
    eval_delta: dict[str, Any],
    model_delta: dict[str, Any],
    hybrid_delta: dict[str, Any],
    contribution_delta: dict[str, Any],
    item_delta: dict[str, Any],
    failure_cases: dict[str, Any],
) -> dict[str, Any]:
    common_samples = int(sample_overlap.get("common_samples") or 0)
    model_total = float(eval_delta.get("f1") or 0.0)
    critical_model = [float(((model_delta.get(field) or {}).get("truth_accuracy_delta")) or 0.0) for field in CRITICAL_FIELDS]
    critical_hybrid = [float(((hybrid_delta.get(field) or {}).get("truth_accuracy_delta")) or 0.0) for field in CRITICAL_FIELDS]
    model_use = [float(((contribution_delta.get(field) or {}).get("layoutlm_only_rate_delta")) or 0.0) for field in CRITICAL_FIELDS]
    hybrid_item = float((((item_delta.get("hybrid") or {}).get("rate_delta")) or 0.0))
    model_item = float((((item_delta.get("layoutlm_only") or {}).get("rate_delta")) or 0.0))
    hybrid_regressions = len(failure_cases.get("hybrid_regressions", []) or [])
    model_regressions = len(failure_cases.get("model_regressions", []) or [])

    if common_samples < 25:
        return {
            "category": "checkpoint_still_too_weak_overall",
            "confidence": "low",
            "reason": f"Only {common_samples} common samples available; evidence is too weak for a strong conclusion.",
        }
    if model_total <= 0 and sum(1 for value in critical_model if value > 0) <= 1:
        return {
            "category": "checkpoint_still_too_weak_overall",
            "confidence": "high",
            "reason": "Evaluation F1 and critical-field truth deltas indicate the improved checkpoint did not materially beat baseline.",
        }
    if sum(1 for value in critical_model if value > 0) >= 2 and sum(1 for value in critical_hybrid if value <= 0) >= 2:
        if sum(1 for value in model_use if value <= 0) >= 2:
            return {
                "category": "model_improved_but_hybrid_not_using_it_enough",
                "confidence": "high",
                "reason": "Critical model fields improved, but hybrid gains lag and model provenance did not rise enough on those fields.",
            }
        return {
            "category": "fusion_too_conservative",
            "confidence": "medium",
            "reason": "Model truth deltas are positive on critical fields, but hybrid truth deltas are flat or negative despite some model use increase.",
        }
    if hybrid_item < 0 or model_item < 0:
        return {
            "category": "item_decoding_weak",
            "confidence": "medium",
            "reason": "Item coherence regressed in model or hybrid outputs; item grouping/decoding remains unstable.",
        }
    if model_regressions > 0 and hybrid_regressions > 0 and sum(1 for value in critical_model if value > 0) <= 1:
        return {
            "category": "weak_labels_still_noisy",
            "confidence": "medium",
            "reason": "Both model and hybrid regressions persist without broad critical-field gains, suggesting the training signal is still noisy.",
        }
    return {
        "category": "model_improved_but_hybrid_not_using_it_enough",
        "confidence": "medium",
        "reason": "The model shows some gains, but the hybrid path is not translating them consistently into better final outputs.",
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Diagnosis Report\n\n")
    headline = report.get("headline", {}) or {}
    lines.append(f"- Improved model beat baseline: `{headline.get('improved_model_beats_baseline')}`\n")
    lines.append(f"- Hybrid improved: `{headline.get('hybrid_improved_over_baseline')}`\n")
    lines.append(f"- Model contribution increased on critical fields: `{headline.get('model_contribution_increased_on_critical_fields')}`\n\n")

    bottleneck = report.get("bottleneck_diagnosis", {}) or {}
    lines.append("## Bottleneck Diagnosis\n\n")
    lines.append(f"- Category: `{bottleneck.get('category')}`\n")
    lines.append(f"- Confidence: `{bottleneck.get('confidence')}`\n")
    lines.append(f"- Reason: {bottleneck.get('reason')}\n\n")

    lines.append("## Critical Fields\n\n")
    for row in ((report.get("critical_field_summary") or {}).get("fields") or []):
        lines.append(
            f"- `{row.get('field')}`: model truth delta={row.get('model_truth_delta')}, "
            f"hybrid truth delta={row.get('hybrid_truth_delta')}, model-use delta={row.get('model_use_delta')}\n"
        )

    lines.append("\n## Secondary Fields\n\n")
    for row in ((report.get("secondary_field_summary") or {}).get("fields") or []):
        lines.append(
            f"- `{row.get('field')}`: model truth delta={row.get('model_truth_delta')}, "
            f"hybrid truth delta={row.get('hybrid_truth_delta')}, model-use delta={row.get('model_use_delta')}\n"
        )

    lines.append("\n## Regressions\n\n")
    regressions = report.get("regression_summary", {}) or {}
    lines.append(f"- Model regressions: {regressions.get('model_regression_count')}\n")
    lines.append(f"- Hybrid regressions: {regressions.get('hybrid_regression_count')}\n")
    for row in regressions.get("top_regressed_fields", []) or []:
        lines.append(f"- `{row.get('field')}`: {row.get('count')}\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
