#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.evaluation.field_comparison import compare_fields
from src.receipt_ai.pipelines.entrypoints import run_easyocr_rules, run_hybrid


CRITICAL_FIELDS = [
    "vendor.name",
    "invoice.date",
    "totals.total",
    "invoice.bill_number",
    "invoice.order_number",
    "invoice.table_number",
    "invoice.cashier",
]


@dataclass(slots=True)
class ModeScore:
    mode: str
    truth_hits: int = 0
    truth_total: int = 0
    coverage_hits: int = 0
    coverage_total: int = 0
    missing_critical: int = 0
    item_coherent: int = 0
    item_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["truth_accuracy"] = round(self.truth_hits / self.truth_total, 4) if self.truth_total > 0 else None
        out["coverage_rate"] = round(self.coverage_hits / self.coverage_total, 4) if self.coverage_total > 0 else None
        out["item_coherence_rate"] = round(self.item_coherent / self.item_total, 4) if self.item_total > 0 else None
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ablation between baseline and tuned rules/hybrid policies.")
    parser.add_argument("--baseline-comparison-file", required=True, help="Existing comparison JSON with current rules/hybrid")
    parser.add_argument("--output-dir", default="outputs/ablation", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap over baseline subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_file = Path(args.baseline_comparison_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = json.loads(baseline_file.read_text(encoding="utf-8"))
    if args.max_samples > 0:
        baseline_rows = baseline_rows[: args.max_samples]

    cfg = ReceiptAIConfig.from_env()

    scores = {
        "current_rules": ModeScore(mode="current_rules"),
        "improved_rules": ModeScore(mode="improved_rules"),
        "current_hybrid": ModeScore(mode="current_hybrid"),
        "tuned_hybrid": ModeScore(mode="tuned_hybrid"),
    }

    per_sample: list[dict[str, Any]] = []

    for row in baseline_rows:
        sample_id = str(row.get("sample_id", ""))
        image_path = row.get("image_path")
        if not image_path:
            continue
        modes = row.get("modes", {})
        gt = row.get("ground_truth")

        current_rules = (modes.get("easyocr_rules") or {}).get("result") or {}
        current_hybrid = (modes.get("hybrid") or {}).get("result") or {}
        if not current_rules or not current_hybrid:
            continue

        improved_rules = run_easyocr_rules(image_path, config=cfg).to_dict()
        tuned_hybrid = run_hybrid(image_path, config=cfg).to_dict()

        variants = {
            "current_rules": current_rules,
            "improved_rules": improved_rules,
            "current_hybrid": current_hybrid,
            "tuned_hybrid": tuned_hybrid,
        }

        sample_out = {"sample_id": sample_id, "critical": {}}

        for mode_name, result in variants.items():
            _update_scores(scores[mode_name], result, gt)
            sample_out["critical"][mode_name] = _critical_fields_snapshot(result)

        per_sample.append(sample_out)

    summary = {
        "baseline_file": str(baseline_file),
        "total_samples": len(per_sample),
        "scores": {name: score.to_dict() for name, score in scores.items()},
        "delta": {
            "rules_truth_accuracy": _delta(scores["improved_rules"], scores["current_rules"], "truth"),
            "hybrid_truth_accuracy": _delta(scores["tuned_hybrid"], scores["current_hybrid"], "truth"),
            "rules_coverage": _delta(scores["improved_rules"], scores["current_rules"], "coverage"),
            "hybrid_coverage": _delta(scores["tuned_hybrid"], scores["current_hybrid"], "coverage"),
            "rules_missing_critical": scores["improved_rules"].missing_critical - scores["current_rules"].missing_critical,
            "hybrid_missing_critical": scores["tuned_hybrid"].missing_critical - scores["current_hybrid"].missing_critical,
        },
    }

    json_path = output_dir / "policy_ablation_report.json"
    md_path = output_dir / "policy_ablation_report.md"
    per_sample_path = output_dir / "policy_ablation_per_sample.json"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_sample_path.write_text(json.dumps(per_sample, indent=2), encoding="utf-8")
    md_path.write_text(_format_markdown(summary), encoding="utf-8")

    print(f"Saved ablation report: {json_path}")
    print(f"Saved ablation markdown: {md_path}")
    print(f"Saved per-sample ablation: {per_sample_path}")


def _update_scores(score: ModeScore, result: dict[str, Any], ground_truth: dict[str, Any] | None) -> None:
    snap = _critical_fields_snapshot(result)

    for field_name in CRITICAL_FIELDS:
        value = str(snap.get(field_name, "") or "").strip()
        score.coverage_total += 1
        if value:
            score.coverage_hits += 1
        else:
            score.missing_critical += 1

    # Item coherence heuristic.
    score.item_total += 1
    if _items_coherent(result):
        score.item_coherent += 1

    if not ground_truth:
        return

    comps = compare_fields(result, result, result, ground_truth)
    by_field = {c.field_name: c for c in comps}
    for field_name in CRITICAL_FIELDS:
        comp = by_field.get(field_name)
        if comp is None or not comp.truth_present:
            continue
        score.truth_total += 1
        if comp.easyocr_vs_truth:
            score.truth_hits += 1


def _critical_fields_snapshot(result: dict[str, Any]) -> dict[str, str]:
    return {
        "vendor.name": str(((result.get("vendor") or {}).get("name") or "")).strip(),
        "invoice.date": str(((result.get("invoice") or {}).get("date") or "")).strip(),
        "totals.total": str(((result.get("totals") or {}).get("total") or "")).strip(),
        "invoice.bill_number": str(((result.get("invoice") or {}).get("bill_number") or "")).strip(),
        "invoice.order_number": str(((result.get("invoice") or {}).get("order_number") or "")).strip(),
        "invoice.table_number": str(((result.get("invoice") or {}).get("table_number") or "")).strip(),
        "invoice.cashier": str(((result.get("invoice") or {}).get("cashier") or "")).strip(),
    }


def _items_coherent(result: dict[str, Any]) -> bool:
    items = (result.get("items") or [])
    if not items:
        return True
    for item in items:
        name = str((item or {}).get("name") or "").strip()
        line_total = float((item or {}).get("line_total") or 0.0)
        if not name or line_total < 0:
            return False
    return True


def _delta(new: ModeScore, old: ModeScore, metric: str) -> float | None:
    if metric == "truth":
        if old.truth_total == 0 or new.truth_total == 0:
            return None
        return round((new.truth_hits / new.truth_total) - (old.truth_hits / old.truth_total), 4)
    if metric == "coverage":
        if old.coverage_total == 0 or new.coverage_total == 0:
            return None
        return round((new.coverage_hits / new.coverage_total) - (old.coverage_hits / old.coverage_total), 4)
    return None


def _format_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Policy Ablation Report\n\n")
    lines.append(f"- Samples: {summary.get('total_samples', 0)}\n")
    lines.append(f"- Baseline: `{summary.get('baseline_file', '')}`\n\n")

    lines.append("## Scores\n\n")
    scores = summary.get("scores", {})
    for mode_name in ["current_rules", "improved_rules", "current_hybrid", "tuned_hybrid"]:
        row = scores.get(mode_name, {})
        lines.append(
            f"- `{mode_name}`: truth_acc={row.get('truth_accuracy')}, coverage={row.get('coverage_rate')}, missing_critical={row.get('missing_critical')}, item_coherence={row.get('item_coherence_rate')}\n"
        )

    lines.append("\n## Deltas\n\n")
    delta = summary.get("delta", {})
    for key in [
        "rules_truth_accuracy",
        "hybrid_truth_accuracy",
        "rules_coverage",
        "hybrid_coverage",
        "rules_missing_critical",
        "hybrid_missing_critical",
    ]:
        lines.append(f"- `{key}`: {delta.get(key)}\n")

    return "".join(lines)


if __name__ == "__main__":
    main()
