#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.receipt_ai.runtime.policy import RuntimePolicy, save_runtime_policy, load_runtime_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote best-policy recommendation to runtime default config.")
    parser.add_argument("--recommendation-json", default="outputs/runtime/best_policy_recommendation.json")
    parser.add_argument("--checkpoint", default="", help="Override preferred checkpoint path")
    parser.add_argument("--policy-path", default="default_config.json")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--fallback-mode", default="easyocr_rules", choices=["easyocr_rules", "layoutlm_only", "hybrid"])
    parser.add_argument("--model-field-confidence", type=float, default=None)
    parser.add_argument("--model-semantic-strong-confidence", type=float, default=None)
    parser.add_argument("--model-takeover-margin", type=float, default=None)
    parser.add_argument("--low-confidence-guard", type=float, default=None)
    parser.add_argument("--total-item-consistency-tolerance", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rec_path = Path(args.recommendation_json).expanduser().resolve()
    if not rec_path.exists():
        raise FileNotFoundError(f"Recommendation file not found: {rec_path}")

    recommendation = json.loads(rec_path.read_text(encoding="utf-8"))
    old_policy = load_runtime_policy(args.policy_path)

    preferred_checkpoint = args.checkpoint or old_policy.preferred_checkpoint
    thresholds = dict(old_policy.hybrid_thresholds)
    override_map = {
        "model_field_confidence": args.model_field_confidence,
        "model_semantic_strong_confidence": args.model_semantic_strong_confidence,
        "model_takeover_margin": args.model_takeover_margin,
        "low_confidence_guard": args.low_confidence_guard,
        "total_item_consistency_tolerance": args.total_item_consistency_tolerance,
    }
    for key, value in override_map.items():
        if value is not None:
            thresholds[key] = float(value)

    recommended_mode = str(recommendation.get("recommended_default_mode", old_policy.default_mode))
    if recommended_mode not in {"easyocr_rules", "layoutlm_only", "hybrid"}:
        recommended_mode = old_policy.default_mode if old_policy.default_mode in {"easyocr_rules", "layoutlm_only", "hybrid"} else "hybrid"

    policy = RuntimePolicy(
        default_mode=recommended_mode,
        preferred_checkpoint=preferred_checkpoint,
        fallback_mode_on_model_failure=args.fallback_mode,
        allow_legacy_checkpoint=bool(args.allow_legacy_checkpoint),
        hybrid_thresholds=thresholds,
        output=dict(old_policy.output),
        deterministic_seed=int(old_policy.deterministic_seed),
        decision=dict(old_policy.decision),
        evidence={
            "recommendation_file": str(rec_path),
            "confidence": recommendation.get("evidence", {}).get("confidence"),
            "sample_size": recommendation.get("evidence", {}).get("total_samples"),
        },
    )

    if args.dry_run:
        print(json.dumps(policy.to_dict(), indent=2))
        return

    policy_target = Path(args.policy_path).expanduser().resolve()
    if policy_target.exists():
        backup_dir = policy_target.parent / "history"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{policy_target.stem}_{stamp}.json"
        backup_path.write_text(policy_target.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backed up previous policy to: {backup_path}")

    policy_path = save_runtime_policy(policy, args.policy_path)
    print(f"Promoted runtime policy: {policy_path}")


if __name__ == "__main__":
    main()
