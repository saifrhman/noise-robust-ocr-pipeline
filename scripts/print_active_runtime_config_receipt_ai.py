#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.receipt_ai.runtime.policy import load_runtime_policy
from src.receipt_ai.model.checkpoint_registry import load_checkpoint_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print active runtime policy and checkpoint registry summary.")
    parser.add_argument("--policy-path", default="")
    parser.add_argument("--registry-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_runtime_policy(args.policy_path or None)
    registry = load_checkpoint_registry(args.registry_path or None)

    payload = {
        "runtime_policy": policy.to_dict(),
        "registry_summary": {
            "active_default_checkpoint": (registry or {}).get("active_default_checkpoint") if registry else "",
            "last_trained_checkpoint": (registry or {}).get("last_trained_checkpoint") if registry else "",
            "best_validated_checkpoint": (registry or {}).get("best_validated_checkpoint") if registry else "",
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
