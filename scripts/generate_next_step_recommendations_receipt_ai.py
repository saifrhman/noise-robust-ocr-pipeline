#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ranked next-step recommendations from experiment artifacts.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_root / "diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(experiment_root / "report" / "experiment_summary.json")
    diagnosis = _read_json(output_dir / "diagnosis_report.json") if (output_dir / "diagnosis_report.json").exists() else {}
    failure_cases = _read_json(experiment_root / "report" / "failure_cases.json")

    recommendations = _build_recommendations(summary, diagnosis, failure_cases)

    json_path = output_dir / "next_step_recommendations.json"
    md_path = output_dir / "next_step_recommendations.md"
    json_path.write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(recommendations), encoding="utf-8")

    print(f"Saved recommendations JSON: {json_path}")
    print(f"Saved recommendations Markdown: {md_path}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_recommendations(summary: dict[str, Any], diagnosis: dict[str, Any], failure_cases: dict[str, Any]) -> dict[str, Any]:
    bottleneck = ((diagnosis.get("bottleneck_diagnosis") or {}).get("category")) or ""
    evaluation_delta = ((summary.get("evaluation") or {}).get("delta")) or {}
    model_delta = ((((summary.get("comparison") or {}).get("model_field_metrics") or {}).get("delta")) or {})
    hybrid_delta = ((((summary.get("comparison") or {}).get("hybrid_field_metrics") or {}).get("delta")) or {})
    contribution_delta = (((summary.get("model_contribution") or {}).get("delta")) or {})
    item_delta = ((((summary.get("comparison") or {}).get("item_coherence") or {}).get("delta")) or {})

    recs: list[dict[str, Any]] = []

    def add(category: str, reason: str, expected_impact: str, affected_fields: list[str], confidence: str, priority: int) -> None:
        recs.append(
            {
                "category": category,
                "reason": reason,
                "expected_impact": expected_impact,
                "affected_fields": affected_fields,
                "confidence": confidence,
                "priority": priority,
            }
        )

    if bottleneck in {"weak_labels_still_noisy", "checkpoint_still_too_weak_overall"}:
        add(
            "weak-label alignment fixes",
            "Critical-field gains are limited and regressions remain, which points to training signal quality still being a bottleneck.",
            "Cleaner supervision should improve vendor/date/total recall and reduce inconsistent regressions.",
            ["vendor.name", "invoice.date", "totals.total", "invoice.bill_number", "invoice.order_number", "invoice.table_number"],
            "medium" if bottleneck == "weak_labels_still_noisy" else "high",
            1,
        )

    if bottleneck in {"fusion_too_conservative", "model_improved_but_hybrid_not_using_it_enough"}:
        add(
            "fusion threshold tuning",
            "Model field deltas are positive, but hybrid gains or model provenance increases are weaker than expected.",
            "Better hybrid adoption of correct model fields should improve final vendor/date/total output quality.",
            ["vendor.name", "invoice.date", "totals.total"],
            "high",
            1,
        )

    if float((((item_delta.get("hybrid") or {}).get("rate_delta")) or 0.0)) < 0 or float((((item_delta.get("layoutlm_only") or {}).get("rate_delta")) or 0.0)) < 0:
        add(
            "item decoder fixes",
            "Item coherence regressed in model or hybrid output; item decoding remains heuristic and unstable.",
            "Cleaner item grouping should reduce item collapse and improve totals consistency.",
            ["items", "totals.total"],
            "medium",
            2,
        )

    highlight_fields = ["vendor.name", "invoice.date", "totals.total"]
    if any(float(((model_delta.get(field) or {}).get("truth_accuracy_delta")) or 0.0) > 0 for field in highlight_fields):
        add(
            "more training / longer epochs",
            "The checkpoint shows at least some positive movement on critical fields, which suggests additional training may convert partial gains into more stable improvements.",
            "Could lift critical-field F1 and reduce variance across samples.",
            highlight_fields,
            "medium",
            3,
        )

    if any(float(((contribution_delta.get(field) or {}).get("layoutlm_only_rate_delta")) or 0.0) < 0 for field in highlight_fields):
        add(
            "parser fixes",
            "Hybrid still leans back to rules or non-model provenance on some critical fields, indicating rule outputs may still dominate specific failure patterns.",
            "Tightening parser coverage can reduce incorrect fallback behavior and improve hybrid consistency.",
            ["invoice.bill_number", "invoice.order_number", "invoice.table_number"],
            "low",
            4,
        )

    if float(evaluation_delta.get("f1") or 0.0) > 0 and len(failure_cases.get("hybrid_regressions", []) or []) == 0:
        add(
            "checkpoint replacement/promotion",
            "The improved checkpoint appears better overall and does not show meaningful hybrid regressions in the saved failure cases.",
            "Could simplify rollout by making the stronger checkpoint the default.",
            highlight_fields,
            "low",
            5,
        )

    recs.sort(key=lambda row: (row["priority"], row["category"]))
    return {"recommendations": recs}


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Next-Step Recommendations\n\n")
    for row in payload.get("recommendations", []) or []:
        lines.append(f"## {row.get('category').title()}\n\n")
        lines.append(f"- Priority: {row.get('priority')}\n")
        lines.append(f"- Confidence: `{row.get('confidence')}`\n")
        lines.append(f"- Affected fields: {', '.join(row.get('affected_fields') or [])}\n")
        lines.append(f"- Reason: {row.get('reason')}\n")
        lines.append(f"- Expected impact: {row.get('expected_impact')}\n\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
