from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any

from src.receipt_ai.evaluation.field_comparison import compare_fields


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
class ImprovementReport:
    comparison_file: str
    analysis_dir: str
    total_samples: int
    field_wins: dict[str, dict[str, int]]
    top_error_buckets: list[dict[str, Any]]
    top_disagreement_patterns: list[dict[str, Any]]
    critical_failure_cases: dict[str, list[dict[str, Any]]]
    item_coherence_failures: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ErrorDrivenImprovementReporter:
    """Builds a concise improvement plan from comparison + analysis artifacts."""

    def __init__(self, comparison_file: Path, analysis_dir: Path) -> None:
        self.comparison_file = comparison_file
        self.analysis_dir = analysis_dir

    def generate(self) -> ImprovementReport:
        comparison_data = json.loads(self.comparison_file.read_text(encoding="utf-8"))
        error_summary = self._read_json_optional(self.analysis_dir / "error_summary.json")
        top_disagreements = self._read_json_optional(self.analysis_dir / "top_disagreements.json")
        item_analyses = self._read_json_optional(self.analysis_dir / "item_analyses.json")

        field_wins = self._compute_field_wins(comparison_data)
        top_error_buckets = self._compute_top_error_buckets(error_summary)
        top_disagreement_patterns = self._compute_top_disagreement_patterns(top_disagreements)
        critical_failure_cases = self._compute_critical_failure_cases(top_disagreements)
        item_coherence_failures = self._compute_item_coherence_failures(item_analyses)

        return ImprovementReport(
            comparison_file=str(self.comparison_file),
            analysis_dir=str(self.analysis_dir),
            total_samples=len(comparison_data),
            field_wins=field_wins,
            top_error_buckets=top_error_buckets,
            top_disagreement_patterns=top_disagreement_patterns,
            critical_failure_cases=critical_failure_cases,
            item_coherence_failures=item_coherence_failures,
        )

    def to_markdown(self, report: ImprovementReport) -> str:
        lines: list[str] = []
        lines.append("# Error-Driven Improvement Report\n\n")
        lines.append(f"- Comparison file: `{report.comparison_file}`\n")
        lines.append(f"- Analysis directory: `{report.analysis_dir}`\n")
        lines.append(f"- Total samples: {report.total_samples}\n\n")

        lines.append("## Field Winners\n\n")
        for field_name, counters in sorted(report.field_wins.items()):
            lines.append(
                f"- `{field_name}`: rules={counters.get('easyocr_rules', 0)}, model={counters.get('layoutlm_only', 0)}, hybrid={counters.get('hybrid', 0)}\n"
            )

        lines.append("\n## Recurring Error Buckets\n\n")
        if report.top_error_buckets:
            for bucket in report.top_error_buckets:
                lines.append(f"- `{bucket['bucket']}`: {bucket['count']}\n")
        else:
            lines.append("- No bucket summary found.\n")

        lines.append("\n## Top Disagreement Patterns\n\n")
        if report.top_disagreement_patterns:
            for row in report.top_disagreement_patterns:
                lines.append(
                    f"- `{row['disagreement_type']}` on `{row['field_name']}`: {row['count']}\n"
                )
        else:
            lines.append("- No disagreement patterns found.\n")

        lines.append("\n## Critical Field Failure Cases\n\n")
        for key in [
            "vendor",
            "date",
            "total",
            "bill_order_table",
            "cashier",
        ]:
            rows = report.critical_failure_cases.get(key, [])
            lines.append(f"### {key.replace('_', ' ').title()}\n\n")
            if not rows:
                lines.append("- No failures found in artifacts.\n")
                continue
            for row in rows[:8]:
                lines.append(
                    f"- sample `{row['sample_id']}` field `{row['field_name']}`: {row['disagreement_type']} ({row['severity']})\n"
                )

        lines.append("\n## Item Coherence Failures (Heuristic)\n\n")
        if report.item_coherence_failures:
            for row in report.item_coherence_failures[:12]:
                lines.append(
                    f"- sample `{row['sample_id']}` mode `{row['mode']}`: {row['reason']}\n"
                )
        else:
            lines.append("- No item coherence failures flagged in current artifacts.\n")

        lines.append("\n## Recommended Next Improvements\n\n")
        lines.append("- Tighten parser extraction for bill/order/table and payment rows where all modes are empty or conflicting.\n")
        lines.append("- Keep hybrid conservative for low-confidence model semantic fields and preserve rule totals by default.\n")
        lines.append("- Re-run weak-label analysis after alignment updates and compare summary deltas.\n")
        lines.append("- Treat item analysis as heuristic; use flagged receipts for manual inspection before rule hardening.\n")
        return "".join(lines)

    def _compute_field_wins(self, comparison_data: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        wins: dict[str, Counter[str]] = defaultdict(Counter)
        for sample in comparison_data:
            modes = sample.get("modes", {})
            rules = modes.get("easyocr_rules", {}).get("result") or {}
            model = modes.get("layoutlm_only", {}).get("result") or {}
            hybrid = modes.get("hybrid", {}).get("result") or {}
            truth = sample.get("ground_truth")
            if not rules or not model or not hybrid:
                continue

            comps = compare_fields(rules, model, hybrid, truth)
            for comp in comps:
                # Gold-backed winner when available.
                if comp.truth_present:
                    if comp.easyocr_vs_truth:
                        wins[comp.field_name]["easyocr_rules"] += 1
                    if comp.layoutlm_vs_truth:
                        wins[comp.field_name]["layoutlm_only"] += 1
                    if comp.hybrid_vs_truth:
                        wins[comp.field_name]["hybrid"] += 1
                    continue

                # Heuristic winner: only one mode has a non-empty value.
                present = {
                    "easyocr_rules": comp.easyocr_present,
                    "layoutlm_only": comp.layoutlm_present,
                    "hybrid": comp.hybrid_present,
                }
                present_modes = [name for name, ok in present.items() if ok]
                if len(present_modes) == 1:
                    wins[comp.field_name][present_modes[0]] += 1
                elif comp.hybrid_present and (not comp.easyocr_present or not comp.layoutlm_present):
                    wins[comp.field_name]["hybrid"] += 1

        out: dict[str, dict[str, int]] = {}
        for field_name, counter in wins.items():
            out[field_name] = {
                "easyocr_rules": int(counter.get("easyocr_rules", 0)),
                "layoutlm_only": int(counter.get("layoutlm_only", 0)),
                "hybrid": int(counter.get("hybrid", 0)),
            }
        return out

    def _compute_top_error_buckets(self, error_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not error_summary:
            return []
        buckets = error_summary.get("bucket_counts", {}) or {}
        rows = [{"bucket": key, "count": int(val)} for key, val in buckets.items()]
        rows.sort(key=lambda x: x["count"], reverse=True)
        return rows[:10]

    def _compute_top_disagreement_patterns(self, top_disagreements: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not top_disagreements:
            return []
        counter: Counter[tuple[str, str]] = Counter()
        for row in top_disagreements:
            counter[(str(row.get("disagreement_type", "")), str(row.get("field_name", "")))] += 1
        rows = [
            {
                "disagreement_type": k[0],
                "field_name": k[1],
                "count": int(v),
            }
            for k, v in counter.items()
            if k[0] and k[1]
        ]
        rows.sort(key=lambda x: x["count"], reverse=True)
        return rows[:15]

    def _compute_critical_failure_cases(self, top_disagreements: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {
            "vendor": [],
            "date": [],
            "total": [],
            "bill_order_table": [],
            "cashier": [],
        }
        if not top_disagreements:
            return out

        for row in top_disagreements:
            field_name = str(row.get("field_name", ""))
            if field_name == "vendor.name" or field_name == "vendor.address":
                out["vendor"].append(row)
            elif field_name == "invoice.date":
                out["date"].append(row)
            elif field_name == "totals.total":
                out["total"].append(row)
            elif field_name in {"invoice.bill_number", "invoice.order_number", "invoice.table_number"}:
                out["bill_order_table"].append(row)
            elif field_name == "invoice.cashier":
                out["cashier"].append(row)

        for key in out:
            out[key] = sorted(out[key], key=lambda x: str(x.get("severity", "")))
        return out

    def _compute_item_coherence_failures(self, item_analyses: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not item_analyses:
            return []
        rows: list[dict[str, Any]] = []
        analyses = item_analyses.get("analyses", []) or []
        for row in analyses:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                continue
            for mode in ["easyocr", "layoutlm", "hybrid"]:
                coherent = bool(row.get(f"{mode}_items_coherent", True))
                item_count = int(row.get(f"{mode}_item_count", 0))
                if not coherent:
                    rows.append({"sample_id": sample_id, "mode": mode, "reason": "incoherent_item_rows"})
                if item_count == 0 and any("item" in str(note).lower() for note in row.get("notes", [])):
                    rows.append({"sample_id": sample_id, "mode": mode, "reason": "missing_items_with_item_notes"})

            for note in row.get("notes", []) or []:
                text = str(note)
                if "implied_total" in text.lower() or "diverges" in text.lower():
                    rows.append({"sample_id": sample_id, "mode": "hybrid", "reason": text})

        return rows

    @staticmethod
    def _read_json_optional(path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
