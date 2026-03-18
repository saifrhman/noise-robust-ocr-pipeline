#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket experiment failure cases into practical diagnosis groups.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else experiment_root / "diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_cases = _read_json(experiment_root / "report" / "failure_cases.json")
    buckets = _bucket_failures(failure_cases)

    json_path = output_dir / "failure_buckets.json"
    md_path = output_dir / "failure_buckets.md"
    json_path.write_text(json.dumps(buckets, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(buckets), encoding="utf-8")

    print(f"Saved failure buckets JSON: {json_path}")
    print(f"Saved failure buckets Markdown: {md_path}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _bucket_failures(payload: dict[str, Any]) -> dict[str, Any]:
    groups = {
        "vendor_missing_in_model": [],
        "date_normalized_wrong": [],
        "total_mismatch": [],
        "hybrid_used_rules_where_model_was_correct": [],
        "hybrid_trusted_model_where_rules_were_better": [],
        "item_grouping_collapse": [],
    }

    for row in payload.get("model_regressions", []) or []:
        fields = set(row.get("regressed_fields", []) or [])
        if "vendor.name" in fields:
            groups["vendor_missing_in_model"].append(row)
        if "invoice.date" in fields:
            groups["date_normalized_wrong"].append(row)
        if "totals.total" in fields:
            groups["total_mismatch"].append(row)
        if row.get("item_coherence_regressed"):
            groups["item_grouping_collapse"].append(row)

    for row in payload.get("hybrid_regressions", []) or []:
        fields = set(row.get("regressed_fields", []) or [])
        if "vendor.name" in fields:
            groups["hybrid_used_rules_where_model_was_correct"].append(row)
        if "invoice.date" in fields or "totals.total" in fields:
            groups["hybrid_trusted_model_where_rules_were_better"].append(row)
        if row.get("item_coherence_regressed"):
            groups["item_grouping_collapse"].append(row)

    return {
        "bucket_counts": {name: len(rows) for name, rows in groups.items()},
        "buckets": groups,
        "heuristic_note": "Buckets are derived from saved regression fields and item-coherence flags; item grouping remains heuristic.",
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Failure Buckets\n\n")
    lines.append(f"- Note: {payload.get('heuristic_note')}\n\n")
    for name, count in (payload.get("bucket_counts") or {}).items():
        lines.append(f"## {name}\n\n")
        lines.append(f"- Count: {count}\n")
        for row in ((payload.get("buckets") or {}).get(name) or [])[:10]:
            lines.append(f"- `{row.get('sample_id')}` fields={row.get('regressed_fields')} item_regressed={row.get('item_coherence_regressed')}\n")
        lines.append("\n")
    return "".join(lines)


if __name__ == "__main__":
    main()
