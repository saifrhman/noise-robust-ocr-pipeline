#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select recommended default extraction policy from artifacts.")
    parser.add_argument("--ablation-report", default="outputs/ablation/policy_ablation_report.json")
    parser.add_argument("--improvement-report", default="outputs/improvement/error_driven_improvement_report.json")
    parser.add_argument("--comparison-file", default="outputs/comparison/comparison_val.json")
    parser.add_argument("--sample-size-threshold", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/runtime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation = _read_json(Path(args.ablation_report))
    improvement = _read_json(Path(args.improvement_report))
    comparison = _read_json(Path(args.comparison_file))

    mode_scores = _collect_mode_scores(ablation)
    layoutlm_score = _compute_layoutlm_score_from_comparison(comparison)
    if layoutlm_score is not None:
        mode_scores["layoutlm_only"] = layoutlm_score
    if not mode_scores:
        raise RuntimeError("No comparable mode scores found in ablation report.")

    ranked = sorted(mode_scores.items(), key=lambda kv: kv[1]["composite"], reverse=True)
    recommended_variant = ranked[0][0]
    recommended_mode = _runtime_mode_from_variant(recommended_variant)

    total_samples = int(ablation.get("total_samples", 0)) if isinstance(ablation, dict) else 0
    caveats: list[str] = []
    confidence = "high"
    comparison_samples = len(comparison) if isinstance(comparison, list) else 0
    if total_samples < args.sample_size_threshold:
        confidence = "low"
        caveats.append(
            f"Evidence is weak due to small ablation sample size ({total_samples} < {args.sample_size_threshold})."
        )
    elif total_samples < args.sample_size_threshold * 2:
        confidence = "medium"
        caveats.append("Evidence is moderate; run larger validation for production confidence.")

    critical_field_strategy = _build_field_strategy(improvement)

    checkpoint_note = _checkpoint_quality_note(improvement, ablation)
    if checkpoint_note:
        caveats.append(checkpoint_note)

    recommendation = {
        "recommended_default_mode": recommended_mode,
        "recommended_policy_variant": recommended_variant,
        "mode_ranking": [
            {
                "mode": mode,
                "runtime_mode": _runtime_mode_from_variant(mode),
                **metrics,
            }
            for mode, metrics in ranked
        ],
        "critical_field_strategy": critical_field_strategy,
        "checkpoint_production_readiness": {
            "status": "provisionally_ok" if confidence in {"high", "medium"} else "needs_more_validation",
            "notes": caveats,
        },
        "item_analysis_note": "Item coherence is heuristic only and not gold truth.",
        "evidence": {
            "ablation_report": str(Path(args.ablation_report).expanduser().resolve()),
            "improvement_report": str(Path(args.improvement_report).expanduser().resolve()),
            "comparison_file": str(Path(args.comparison_file).expanduser().resolve()),
            "total_samples": total_samples,
            "comparison_samples": comparison_samples,
            "confidence": confidence,
        },
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "best_policy_recommendation.json"
    md_path = output_dir / "best_policy_recommendation.md"

    json_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(recommendation), encoding="utf-8")

    print(f"Saved recommendation JSON: {json_path}")
    print(f"Saved recommendation summary: {md_path}")


def _collect_mode_scores(ablation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    row = ablation.get("scores", {}) if isinstance(ablation, dict) else {}
    mapping = {
        "easyocr_rules": "current_rules",
        "improved_rules": "improved_rules",
        "current_hybrid": "current_hybrid",
        "tuned_hybrid": "tuned_hybrid",
    }
    for mode, key in mapping.items():
        src = row.get(key) or {}
        if not src:
            continue
        truth = src.get("truth_accuracy")
        coverage = src.get("coverage_rate")
        missing = float(src.get("missing_critical", 0))
        item_coh = src.get("item_coherence_rate")
        truth_val = float(truth) if isinstance(truth, (int, float)) else 0.0
        coverage_val = float(coverage) if isinstance(coverage, (int, float)) else 0.0
        item_val = float(item_coh) if isinstance(item_coh, (int, float)) else 0.0
        composite = (0.6 * truth_val) + (0.3 * coverage_val) + (0.1 * item_val) - (0.02 * missing)
        scores[mode] = {
            "truth_accuracy": truth,
            "coverage_rate": coverage,
            "item_coherence_rate": item_coh,
            "missing_critical": missing,
            "composite": round(composite, 6),
        }

    return scores


def _compute_layoutlm_score_from_comparison(comparison: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(comparison, list) or not comparison:
        return {
            "truth_accuracy": None,
            "coverage_rate": None,
            "item_coherence_rate": None,
            "missing_critical": None,
            "composite": -999.0,
            "note": "No comparison artifact available for layoutlm_only scoring.",
        }

    truth_hits = 0
    truth_total = 0
    coverage_hits = 0
    coverage_total = 0
    missing_critical = 0

    critical_paths = [
        ("vendor", "name"),
        ("invoice", "date"),
        ("totals", "total"),
        ("totals", "subtotal"),
        ("totals", "tax"),
        ("invoice", "bill_number"),
        ("invoice", "order_number"),
        ("invoice", "table_number"),
        ("payment", "method"),
    ]

    for sample in comparison:
        layout = ((sample.get("modes", {}) or {}).get("layoutlm_only", {}) or {}).get("result") or {}
        truth = sample.get("ground_truth") or {}
        if not layout:
            continue
        for parent, child in critical_paths:
            pred = str(((layout.get(parent) or {}).get(child) or "")).strip()
            coverage_total += 1
            if pred:
                coverage_hits += 1
            else:
                missing_critical += 1

            gt_parent = truth.get(parent) if isinstance(truth, dict) else None
            if not isinstance(gt_parent, dict):
                continue
            gt_val = str(gt_parent.get(child) or "").strip()
            if not gt_val:
                continue
            truth_total += 1
            if pred and pred.lower() == gt_val.lower():
                truth_hits += 1

    if coverage_total == 0:
        return None

    truth_accuracy = (truth_hits / truth_total) if truth_total > 0 else None
    coverage_rate = coverage_hits / coverage_total
    composite = (0.6 * float(truth_accuracy or 0.0)) + (0.3 * coverage_rate) - (0.02 * float(missing_critical))
    return {
        "truth_accuracy": round(truth_accuracy, 6) if truth_accuracy is not None else None,
        "coverage_rate": round(coverage_rate, 6),
        "item_coherence_rate": None,
        "missing_critical": float(missing_critical),
        "composite": round(composite, 6),
        "note": "layoutlm_only derived from comparison artifact (heuristic on non-gold fields).",
    }


def _build_field_strategy(improvement: dict[str, Any]) -> dict[str, str]:
    wins = improvement.get("field_wins", {}) if isinstance(improvement, dict) else {}
    strategy: dict[str, str] = {}

    field_groups = {
        "vendor": ["vendor.name", "vendor.address"],
        "date": ["invoice.date", "invoice.time"],
        "total": ["totals.total", "totals.subtotal", "totals.tax"],
        "bill_order_table": ["invoice.bill_number", "invoice.order_number", "invoice.table_number"],
        "payment_method": ["payment.method"],
    }

    for group_name, fields in field_groups.items():
        counters = {"easyocr_rules": 0, "layoutlm_only": 0, "hybrid": 0}
        for field in fields:
            row = wins.get(field) or {}
            for mode in counters:
                counters[mode] += int(row.get(mode, 0))

        winner = max(counters.items(), key=lambda kv: kv[1])[0]
        if winner == "easyocr_rules":
            strategy[group_name] = "rules_preferred"
        elif winner == "layoutlm_only":
            strategy[group_name] = "model_preferred_when_confident"
        else:
            strategy[group_name] = "hybrid_preferred"

    strategy["items"] = "heuristic_item_coherence_only"
    return strategy


def _checkpoint_quality_note(improvement: dict[str, Any], ablation: dict[str, Any]) -> str:
    buckets = improvement.get("top_error_buckets", []) if isinstance(improvement, dict) else []
    for bucket in buckets:
        if str(bucket.get("bucket")) == "schema_reduced_warning" and int(bucket.get("count", 0)) > 0:
            return "Legacy/reduced-schema warnings exist; validate richer-schema checkpoint before full production default."

    scores = ablation.get("scores", {}) if isinstance(ablation, dict) else {}
    tuned = scores.get("tuned_hybrid", {}) if isinstance(scores, dict) else {}
    truth = tuned.get("truth_accuracy")
    if isinstance(truth, (int, float)) and truth < 0.6:
        return "Tuned hybrid truth accuracy is still low; continue checkpoint tuning before strict production rollout."
    return ""


def _to_markdown(rec: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Best Policy Recommendation\n\n")
    lines.append(f"- Recommended default mode: `{rec.get('recommended_default_mode')}`\n")
    lines.append(f"- Recommended policy variant: `{rec.get('recommended_policy_variant')}`\n")
    evidence = rec.get("evidence", {})
    lines.append(f"- Evidence confidence: `{evidence.get('confidence')}`\n")
    lines.append(f"- Samples: {evidence.get('total_samples')}\n\n")

    lines.append("## Mode Ranking\n\n")
    for row in rec.get("mode_ranking", []):
        lines.append(
            f"- `{row.get('mode')}`: composite={row.get('composite')}, truth={row.get('truth_accuracy')}, coverage={row.get('coverage_rate')}, missing_critical={row.get('missing_critical')}\n"
        )

    lines.append("\n## Field Handling Strategy\n\n")
    for field, strategy in rec.get("critical_field_strategy", {}).items():
        lines.append(f"- `{field}` -> `{strategy}`\n")

    lines.append("\n## Checkpoint Readiness\n\n")
    ready = rec.get("checkpoint_production_readiness", {})
    lines.append(f"- Status: `{ready.get('status')}`\n")
    for note in ready.get("notes", []):
        lines.append(f"- {note}\n")

    lines.append("\n## Caveat\n\n")
    lines.append(f"- {rec.get('item_analysis_note')}\n")
    return "".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    p = path.expanduser().resolve()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _runtime_mode_from_variant(variant: str) -> str:
    mapping = {
        "current_rules": "easyocr_rules",
        "improved_rules": "easyocr_rules",
        "easyocr_rules": "easyocr_rules",
        "layoutlm_only": "layoutlm_only",
        "current_hybrid": "hybrid",
        "tuned_hybrid": "hybrid",
        "hybrid": "hybrid",
    }
    return mapping.get(variant, "hybrid")


if __name__ == "__main__":
    main()
