from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.pipelines.entrypoints import run_easyocr_rules, run_hybrid, run_layoutlm_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch receipt extraction and save one JSON per image.")
    parser.add_argument("--mode", default="hybrid", choices=["easyocr_rules", "layoutlm_only", "hybrid"])
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", default="outputs/receipt_ai_batch")
    parser.add_argument("--output-jsonl", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()

    loader = SROIEDatasetLoader(cfg.paths.data_root)
    output_dir = Path(args.output_dir).expanduser().resolve() / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = {
        "easyocr_rules": run_easyocr_rules,
        "layoutlm_only": run_layoutlm_only,
        "hybrid": run_hybrid,
    }[args.mode]

    aggregate: list[str] = []
    processed = 0
    for sample in loader.iter_samples(args.split, val_ratio=args.val_ratio, seed=args.seed, strict=args.strict):
        if args.max_samples > 0 and processed >= args.max_samples:
            break

        payload: dict[str, object]
        try:
            result = runner(sample.image_path, config=cfg)
            payload = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "result": result.to_dict(),
            }
        except Exception as exc:
            if args.strict:
                raise
            payload = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
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


if __name__ == "__main__":
    main()