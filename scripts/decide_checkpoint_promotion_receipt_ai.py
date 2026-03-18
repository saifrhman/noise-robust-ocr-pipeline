#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CRITICAL_FIELDS = ["vendor.name", "invoice.date", "totals.total"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether an improved checkpoint should be promoted.")
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

    decision = _build_decision(summary, diagnosis, failure_cases)

    json_path = output_dir / "checkpoint_promotion_decision.json"
    md_path = output_dir / "checkpoint_promotion_decision.md"
    json_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(decision), encoding="utf-8")

    print(f"Saved promotion decision JSON: {json_path}")
    print(f"Saved promotion decision Markdown: {md_path}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_decision(summary: dict[str, Any], diagnosis: dict[str, Any], failure_cases: dict[str, Any]) -> dict[str, Any]:
    sample_overlap = summary.get("sample_overlap", {}) or {}
    common_samples = int(sample_overlap.get("common_samples") or 0)
    eval_delta = ((summary.get("evaluation") or {}).get("delta")) or {}
    model_delta = ((((summary.get("comparison") or {}).get("model_field_metrics") or {}).get("delta")) or {})
    hybrid_delta = ((((summary.get("comparison") or {}).get("hybrid_field_metrics") or {}).get("delta")) or {})
    contribution_delta = (((summary.get("model_contribution") or {}).get("delta")) or {})
    bottleneck = ((diagnosis.get("bottleneck_diagnosis") or {}).get("category")) or ""

    critical_model_gains = sum(1 for field in CRITICAL_FIELDS if float(((model_delta.get(field) or {}).get("truth_accuracy_delta")) or 0.0) > 0)
    critical_hybrid_gains = sum(1 for field in CRITICAL_FIELDS if float(((hybrid_delta.get(field) or {}).get("truth_accuracy_delta")) or 0.0) > 0)
    critical_model_use_gains = sum(1 for field in CRITICAL_FIELDS if float(((contribution_delta.get(field) or {}).get("layoutlm_only_rate_delta")) or 0.0) > 0)
    hybrid_regressions = len(failure_cases.get("hybrid_regressions", []) or [])

    if common_samples < 25:
        status = "remain_experimental"
        rationale = f"Only {common_samples} common samples were evaluated, so evidence is too weak for promotion."
        confidence = "low"
    elif float(eval_delta.get("f1") or 0.0) > 0 and critical_model_gains >= 2 and critical_hybrid_gains >= 2 and hybrid_regressions == 0:
        status = "replace_current_default"
        rationale = "The improved checkpoint beats baseline overall, improves critical fields, and does not show material hybrid regressions."
        confidence = "medium"
    elif float(eval_delta.get("f1") or 0.0) <= 0 and critical_model_gains <= 1:
        status = "reject"
        rationale = "The improved checkpoint does not show enough overall or critical-field improvement to justify promotion."
        confidence = "high"
    else:
        status = "remain_experimental"
        rationale = "Results are mixed: the checkpoint shows some gains, but evidence is not strong enough for default promotion."
        confidence = "medium"

    return {
        "decision": status,
        "confidence": confidence,
        "rationale": rationale,
        "evidence": {
            "common_samples": common_samples,
            "eval_f1_delta": eval_delta.get("f1"),
            "critical_model_gains": critical_model_gains,
            "critical_hybrid_gains": critical_hybrid_gains,
            "critical_model_use_gains": critical_model_use_gains,
            "hybrid_regressions": hybrid_regressions,
            "diagnosed_bottleneck": bottleneck,
        },
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Checkpoint Promotion Decision\n\n")
    lines.append(f"- Decision: `{payload.get('decision')}`\n")
    lines.append(f"- Confidence: `{payload.get('confidence')}`\n")
    lines.append(f"- Rationale: {payload.get('rationale')}\n\n")
    lines.append("## Evidence\n\n")
    for key, value in (payload.get("evidence") or {}).items():
        lines.append(f"- `{key}`: {value}\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
