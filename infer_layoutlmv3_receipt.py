#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.receipt_ai.runtime.output import format_result_output
from src.receipt_ai.runtime.runner import load_runtime_config, resolve_mode, run_extraction


warnings.warn(
    "infer_layoutlmv3_receipt.py is deprecated. Use run_receipt_ai.py for the production entrypoint.",
    DeprecationWarning,
    stacklevel=2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated compatibility wrapper around the unified receipt-ai runtime.")
    parser.add_argument("--image", required=True, help="Path to receipt image")
    parser.add_argument("--checkpoint", default="", help="Optional fine-tuned LayoutLMv3 checkpoint override")
    parser.add_argument(
        "--extraction-mode",
        default="hybrid",
        choices=["easyocr_rules", "layoutlm_only", "hybrid"],
        help="Extraction mode",
    )
    parser.add_argument("--config", default="", help="Optional runtime config path")
    parser.add_argument("--output-mode", default="full", choices=["full", "minimal"])
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg, policy = load_runtime_config(args.config or None)
    effective_mode, _, messages = resolve_mode(
        args.extraction_mode,
        checkpoint_override=args.checkpoint,
        cfg=cfg,
        policy=policy,
    )
    result = run_extraction(image_path, mode=effective_mode, cfg=cfg)
    payload = format_result_output(result, output_mode=args.output_mode)
    if messages:
        payload["metadata"]["warnings"] = list(payload["metadata"].get("warnings") or []) + list(messages)

    print(json.dumps(payload, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
