from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.runtime.output import format_result_output
from src.receipt_ai.runtime.runner import load_runtime_config, resolve_mode, run_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch receipt extraction and save one JSON per image.")
    parser.add_argument("--mode", default="auto", choices=["auto", "easyocr_rules", "layoutlm_only", "hybrid"])
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", default="outputs/receipt_ai_batch")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-mode", default="", choices=["", "full", "minimal"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, policy = load_runtime_config(args.config or None)
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()
    output_mode = args.output_mode or str((policy.output or {}).get("mode", "full"))
    if output_mode not in {"full", "minimal"}:
        output_mode = "full"

    loader = SROIEDatasetLoader(cfg.paths.data_root)

    requested_mode = policy.default_mode if args.mode == "auto" else args.mode
    effective_mode, checkpoint_used, mode_messages = resolve_mode(
        requested_mode,
        checkpoint_override=args.checkpoint,
        cfg=cfg,
        policy=policy,
    )
    mode_warning = " ".join(mode_messages).strip()

    output_dir = Path(args.output_dir).expanduser().resolve() / effective_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate: list[str] = []
    processed = 0
    for sample in loader.iter_samples(args.split, val_ratio=args.val_ratio, seed=args.seed, strict=args.strict):
        if args.max_samples > 0 and processed >= args.max_samples:
            break

        payload: dict[str, object]
        try:
            result = run_extraction(sample.image_path, mode=effective_mode, cfg=cfg)
            payload = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "requested_mode": requested_mode,
                "effective_mode": effective_mode,
                "checkpoint_used": checkpoint_used,
                "mode_warning": mode_warning,
                "result": format_result_output(
                    result,
                    output_mode=output_mode,
                    include_confidence=bool((policy.output or {}).get("include_confidence", True)),
                    include_provenance=bool((policy.output or {}).get("include_provenance", True)),
                ),
            }
        except Exception as exc:
            if args.strict:
                raise
            payload = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "requested_mode": requested_mode,
                "effective_mode": effective_mode,
                "checkpoint_used": checkpoint_used,
                "mode_warning": mode_warning,
                "error": str(exc),
            }

        (output_dir / f"{sample.sample_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        aggregate.append(json.dumps(payload, ensure_ascii=True))
        processed += 1

    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl).expanduser().resolve()
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl.write_text("\n".join(aggregate) + ("\n" if aggregate else ""), encoding="utf-8")

    print(f"Saved {processed} batch outputs to {output_dir}")
    if mode_warning:
        print(f"Mode warning: {mode_warning}")


if __name__ == "__main__":
    main()
