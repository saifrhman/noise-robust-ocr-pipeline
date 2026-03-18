#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare previous vs latest recommendation artifacts for regressions.")
    parser.add_argument("--previous-recommendation", required=True)
    parser.add_argument("--latest-recommendation", required=True)
    parser.add_argument("--previous-ablation", default="")
    parser.add_argument("--latest-ablation", default="")
    parser.add_argument("--output-dir", default="outputs/runtime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prev = _read_json(Path(args.previous_recommendation))
    latest = _read_json(Path(args.latest_recommendation))
    prev_ab = _read_json(Path(args.previous_ablation)) if args.previous_ablation else {}
    latest_ab = _read_json(Path(args.latest_ablation)) if args.latest_ablation else {}

    report = {
        "recommended_mode_changed": prev.get("recommended_default_mode") != latest.get("recommended_default_mode"),
        "previous_mode": prev.get("recommended_default_mode"),
        "latest_mode": latest.get("recommended_default_mode"),
        "critical_field_strategy_changes": _strategy_changes(prev, latest),
        "critical_field_improvements": _mode_improvements(prev_ab, latest_ab),
        "mode_score_regressions": _mode_regressions(prev_ab, latest_ab),
        "unstable_fields": _unstable_fields(prev, latest),
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "policy_regression_report.json"
    md_path = output_dir / "policy_regression_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    print(f"Saved regression report JSON: {json_path}")
    print(f"Saved regression report summary: {md_path}")


def _strategy_changes(prev: dict[str, Any], latest: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    prev_map = prev.get("critical_field_strategy", {}) if isinstance(prev, dict) else {}
    latest_map = latest.get("critical_field_strategy", {}) if isinstance(latest, dict) else {}
    for key in sorted(set(prev_map.keys()) | set(latest_map.keys())):
        if prev_map.get(key) != latest_map.get(key):
            out.append({"field_group": key, "previous": str(prev_map.get(key)), "latest": str(latest_map.get(key))})
    return out


def _mode_regressions(prev_ab: dict[str, Any], latest_ab: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not prev_ab or not latest_ab:
        return out

    prev_scores = prev_ab.get("scores", {}) if isinstance(prev_ab, dict) else {}
    latest_scores = latest_ab.get("scores", {}) if isinstance(latest_ab, dict) else {}
    for mode in sorted(set(prev_scores.keys()) & set(latest_scores.keys())):
        p = prev_scores.get(mode, {})
        l = latest_scores.get(mode, {})
        p_truth = p.get("truth_accuracy")
        l_truth = l.get("truth_accuracy")
        p_cov = p.get("coverage_rate")
        l_cov = l.get("coverage_rate")
        if isinstance(p_truth, (int, float)) and isinstance(l_truth, (int, float)) and l_truth < p_truth:
            out.append({"mode": mode, "metric": "truth_accuracy", "previous": p_truth, "latest": l_truth, "delta": round(l_truth - p_truth, 6)})
        if isinstance(p_cov, (int, float)) and isinstance(l_cov, (int, float)) and l_cov < p_cov:
            out.append({"mode": mode, "metric": "coverage_rate", "previous": p_cov, "latest": l_cov, "delta": round(l_cov - p_cov, 6)})
    return out


def _mode_improvements(prev_ab: dict[str, Any], latest_ab: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not prev_ab or not latest_ab:
        return out

    prev_scores = prev_ab.get("scores", {}) if isinstance(prev_ab, dict) else {}
    latest_scores = latest_ab.get("scores", {}) if isinstance(latest_ab, dict) else {}
    for mode in sorted(set(prev_scores.keys()) & set(latest_scores.keys())):
        p = prev_scores.get(mode, {})
        l = latest_scores.get(mode, {})
        p_truth = p.get("truth_accuracy")
        l_truth = l.get("truth_accuracy")
        p_cov = p.get("coverage_rate")
        l_cov = l.get("coverage_rate")
        if isinstance(p_truth, (int, float)) and isinstance(l_truth, (int, float)) and l_truth > p_truth:
            out.append({"mode": mode, "metric": "truth_accuracy", "previous": p_truth, "latest": l_truth, "delta": round(l_truth - p_truth, 6)})
        if isinstance(p_cov, (int, float)) and isinstance(l_cov, (int, float)) and l_cov > p_cov:
            out.append({"mode": mode, "metric": "coverage_rate", "previous": p_cov, "latest": l_cov, "delta": round(l_cov - p_cov, 6)})
    return out


def _unstable_fields(prev: dict[str, Any], latest: dict[str, Any]) -> list[str]:
    unstable: list[str] = []
    prev_rank = prev.get("mode_ranking", []) if isinstance(prev, dict) else []
    latest_rank = latest.get("mode_ranking", []) if isinstance(latest, dict) else []
    if prev_rank and latest_rank:
        p_top = [row.get("mode") for row in prev_rank[:2]]
        l_top = [row.get("mode") for row in latest_rank[:2]]
        if p_top != l_top:
            unstable.append("top_mode_order")

    prev_conf = (prev.get("evidence", {}) if isinstance(prev, dict) else {}).get("confidence")
    latest_conf = (latest.get("evidence", {}) if isinstance(latest, dict) else {}).get("confidence")
    if prev_conf != latest_conf:
        unstable.append("evidence_confidence")
    return unstable


def _to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Policy Regression Report\n\n")
    lines.append(f"- Recommended mode changed: `{report.get('recommended_mode_changed')}`\n")
    lines.append(f"- Previous mode: `{report.get('previous_mode')}`\n")
    lines.append(f"- Latest mode: `{report.get('latest_mode')}`\n\n")

    lines.append("## Critical Strategy Changes\n\n")
    changes = report.get("critical_field_strategy_changes", [])
    if not changes:
        lines.append("- No critical field strategy changes.\n")
    for row in changes:
        lines.append(
            f"- `{row.get('field_group')}`: `{row.get('previous')}` -> `{row.get('latest')}`\n"
        )

    lines.append("\n## Regressions\n\n")
    regs = report.get("mode_score_regressions", [])
    if not regs:
        lines.append("- No numeric regressions detected in provided ablation artifacts.\n")
    for row in regs:
        lines.append(
            f"- mode `{row.get('mode')}` metric `{row.get('metric')}`: {row.get('previous')} -> {row.get('latest')} (delta={row.get('delta')})\n"
        )

    lines.append("\n## Improvements\n\n")
    improvements = report.get("critical_field_improvements", [])
    if not improvements:
        lines.append("- No critical-field metric improvements detected in provided ablation artifacts.\n")
    for row in improvements:
        lines.append(
            f"- mode `{row.get('mode')}` metric `{row.get('metric')}`: {row.get('previous')} -> {row.get('latest')} (delta={row.get('delta')})\n"
        )

    lines.append("\n## Unstable Behavior\n\n")
    unstable = report.get("unstable_fields", [])
    if not unstable:
        lines.append("- No instability flags.\n")
    for row in unstable:
        lines.append(f"- `{row}`\n")

    return "".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path:
        return {}
    p = path.expanduser().resolve()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
