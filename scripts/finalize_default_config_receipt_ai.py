#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.receipt_ai.runtime.policy import RuntimePolicy, load_runtime_policy, save_runtime_policy
from src.receipt_ai.model.checkpoint_registry import load_checkpoint_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize repo default_config.json from runtime and experiment artifacts.")
    parser.add_argument("--output-path", default="default_config.json")
    parser.add_argument("--runtime-policy", default="outputs/runtime/recommended_runtime_config.json")
    parser.add_argument("--policy-recommendation", default="outputs/runtime/best_policy_recommendation.json")
    parser.add_argument("--registry-path", default="outputs/runtime/checkpoint_registry.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_policy = load_runtime_policy(args.runtime_policy)
    recommendation = _read_json(Path(args.policy_recommendation))
    registry = load_checkpoint_registry(args.registry_path) or {}
    latest_decision = _find_latest_promotion_decision(Path("outputs"))

    preferred_checkpoint = _choose_checkpoint(latest_decision, runtime_policy, registry)
    default_mode = _choose_default_mode(runtime_policy, recommendation)

    payload = RuntimePolicy(
        default_mode=default_mode,
        preferred_checkpoint=preferred_checkpoint,
        fallback_mode_on_model_failure=runtime_policy.fallback_mode_on_model_failure,
        allow_legacy_checkpoint=runtime_policy.allow_legacy_checkpoint,
        hybrid_thresholds=dict(runtime_policy.hybrid_thresholds),
        output={
            "mode": "full",
            "include_confidence": True,
            "include_provenance": True,
        },
        deterministic_seed=42,
        decision=_decision_summary(latest_decision),
        evidence={
            "runtime_policy_source": str(Path(args.runtime_policy).expanduser().resolve()),
            "policy_recommendation_source": str(Path(args.policy_recommendation).expanduser().resolve()),
            "checkpoint_registry_source": str(Path(args.registry_path).expanduser().resolve()),
        },
    )

    output_path = save_runtime_policy(payload, args.output_path)
    print(f"Saved default config: {output_path}")


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _find_latest_promotion_decision(outputs_root: Path) -> dict[str, Any]:
    matches = sorted(
        outputs_root.glob("**/checkpoint_promotion_decision.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return {}
    payload = _read_json(matches[0])
    payload["_artifact_path"] = str(matches[0].resolve())
    return payload


def _choose_checkpoint(
    decision: dict[str, Any],
    runtime_policy: RuntimePolicy,
    registry: dict[str, Any],
) -> str:
    if decision.get("decision") == "replace_current_default":
        experiment_root = Path(str(decision.get("_artifact_path", ""))).resolve().parents[1]
        summary_path = experiment_root / "report" / "experiment_summary.json"
        summary = _read_json(summary_path)
        improved_checkpoint = str(summary.get("improved_checkpoint", "")).strip()
        if improved_checkpoint:
            return str(Path(improved_checkpoint).expanduser().resolve())

    candidates = [
        str(runtime_policy.preferred_checkpoint or "").strip(),
        str((registry or {}).get("active_default_checkpoint") or "").strip(),
        str((registry or {}).get("best_validated_checkpoint") or "").strip(),
        str((registry or {}).get("last_trained_checkpoint") or "").strip(),
    ]
    for candidate in candidates:
        if candidate:
            return str(Path(candidate).expanduser().resolve())
    return ""


def _choose_default_mode(runtime_policy: RuntimePolicy, recommendation: dict[str, Any]) -> str:
    mode = str(recommendation.get("recommended_default_mode", "")).strip() or runtime_policy.default_mode
    if mode not in {"easyocr_rules", "layoutlm_only", "hybrid"}:
        return "easyocr_rules"
    return mode


def _decision_summary(decision: dict[str, Any]) -> dict[str, Any]:
    if not decision:
        return {
            "status": "no_promotion_artifact_found",
            "confidence": "unknown",
            "rationale": "No checkpoint promotion decision artifact was found. Default config falls back to the current runtime recommendation.",
        }
    return {
        "status": decision.get("decision", ""),
        "confidence": decision.get("confidence", ""),
        "rationale": decision.get("rationale", ""),
        "artifact_path": decision.get("_artifact_path", ""),
    }


if __name__ == "__main__":
    main()
