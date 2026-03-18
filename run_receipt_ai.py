#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.receipt_ai.runtime.output import format_result_output
from src.receipt_ai.runtime.runner import iter_input_images, load_runtime_config, resolve_mode, run_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run receipt extraction on an image or folder using the repo default config.")
    parser.add_argument("input_path", help="Receipt image or folder of receipt images")
    parser.add_argument("--config", default="", help="Optional runtime config path. Defaults to default_config.json")
    parser.add_argument("--mode", default="auto", choices=["auto", "easyocr_rules", "layoutlm_only", "hybrid"])
    parser.add_argument("--checkpoint", default="", help="Optional checkpoint override")
    parser.add_argument("--output", default="", help="Optional output file or directory")
    parser.add_argument("--output-mode", default="", choices=["", "full", "minimal"])
    parser.add_argument("--exclude-confidence", action="store_true")
    parser.add_argument("--exclude-provenance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, policy = load_runtime_config(args.config or None)
    requested_mode = args.mode if args.mode != "auto" else policy.default_mode
    output_mode = args.output_mode or str((policy.output or {}).get("mode", "full"))
    if output_mode not in {"full", "minimal"}:
        output_mode = "full"
    effective_mode, checkpoint_used, messages = resolve_mode(
        requested_mode,
        checkpoint_override=args.checkpoint,
        cfg=cfg,
        policy=policy,
    )

    image_paths = iter_input_images(args.input_path)
    outputs = []
    for image_path in image_paths:
        result = run_extraction(image_path, mode=effective_mode, cfg=cfg)
        payload = format_result_output(
            result,
            output_mode=output_mode,
            include_confidence=bool((policy.output or {}).get("include_confidence", True)) and not args.exclude_confidence,
            include_provenance=bool((policy.output or {}).get("include_provenance", True)) and not args.exclude_provenance,
        )
        outputs.append(
            {
                "image_path": str(image_path),
                "requested_mode": requested_mode,
                "effective_mode": effective_mode,
                "checkpoint_used": checkpoint_used,
                "fallback_messages": messages,
                "result": payload,
            }
        )

    rendered = outputs[0]["result"] if len(outputs) == 1 and not args.output else outputs
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        if len(outputs) == 1:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            for row in outputs:
                sample_path = output_path / f"{Path(row['image_path']).stem}.json"
                sample_path.write_text(json.dumps(row["result"], indent=2), encoding="utf-8")
    else:
        print(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
