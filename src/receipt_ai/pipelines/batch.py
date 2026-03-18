from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.pipelines.entrypoints import run_easyocr_rules, run_hybrid, run_layoutlm_only
from src.receipt_ai.runtime.policy import apply_runtime_policy, load_runtime_policy


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    policy = load_runtime_policy()
    cfg = apply_runtime_policy(cfg, policy)
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()

    loader = SROIEDatasetLoader(cfg.paths.data_root)

    effective_mode = policy.default_mode if args.mode == "auto" else args.mode
    if effective_mode not in {"easyocr_rules", "layoutlm_only", "hybrid"}:
        effective_mode = "hybrid"
    effective_mode, mode_warning = _resolve_mode_with_model_fallback(effective_mode, cfg, policy)

    output_dir = Path(args.output_dir).expanduser().resolve() / effective_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = {
        "easyocr_rules": run_easyocr_rules,
        "layoutlm_only": run_layoutlm_only,
        "hybrid": run_hybrid,
    }[effective_mode]

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
                "requested_mode": args.mode,
                "effective_mode": effective_mode,
                "mode_warning": mode_warning,
                "result": result.to_dict(),
            }
        except Exception as exc:
            if args.strict:
                raise
            payload = {
                "sample_id": sample.sample_id,
                "image_path": str(sample.image_path),
                "requested_mode": args.mode,
                "effective_mode": effective_mode,
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


def _resolve_mode_with_model_fallback(
    requested_mode: str,
    cfg: ReceiptAIConfig,
    policy,
) -> tuple[str, str]:
    if requested_mode not in {"layoutlm_only", "hybrid"}:
        return requested_mode, ""

    checkpoint_path = cfg.paths.model_checkpoint
    if not checkpoint_path.exists():
        fallback = policy.fallback_mode_on_model_failure or "easyocr_rules"
        warning = (
            f"Checkpoint missing at {checkpoint_path}. Falling back from {requested_mode} to {fallback}."
        )
        return fallback, warning

    compatibility = inspect_checkpoint_label_space(checkpoint_path)
    if (not compatibility.is_compatible) and (not compatibility.is_legacy):
        fallback = policy.fallback_mode_on_model_failure or "easyocr_rules"
        warning = (
            f"Checkpoint incompatible ({compatibility.message}). Falling back from {requested_mode} to {fallback}."
        )
        return fallback, warning

    if compatibility.is_legacy and not bool(policy.allow_legacy_checkpoint):
        fallback = policy.fallback_mode_on_model_failure or "easyocr_rules"
        warning = (
            f"Legacy checkpoint detected and legacy usage is disabled. Falling back from {requested_mode} to {fallback}."
        )
        return fallback, warning

    return requested_mode, ""


if __name__ == "__main__":
    main()