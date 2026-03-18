#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.receipt_ai.model.checkpoint_registry import (
    build_checkpoint_registry,
    discover_checkpoints,
    save_checkpoint_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh checkpoint registry and recommended model status.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--registry-path", default="outputs/runtime/checkpoint_registry.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).expanduser().resolve()
    checkpoints = discover_checkpoints(outputs_root)
    registry = build_checkpoint_registry(checkpoints, outputs_root=outputs_root)
    path = save_checkpoint_registry(registry, args.registry_path)

    print(f"Saved checkpoint registry: {path}")
    print(f"Discovered checkpoints: {len(registry.checkpoints)}")
    print(f"Active default checkpoint: {registry.active_default_checkpoint}")
    print(f"Last trained checkpoint: {registry.last_trained_checkpoint}")
    print(f"Best validated checkpoint: {registry.best_validated_checkpoint}")


if __name__ == "__main__":
    main()
