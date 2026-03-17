from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.model.inference import LayoutLMv3TokenClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained receipt checkpoint for schema and inference readiness.")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-json", default="outputs/checkpoint_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    compatibility = inspect_checkpoint_label_space(checkpoint)
    config_payload = _validate_config_mappings(checkpoint)

    loader = SROIEDatasetLoader(cfg.paths.data_root)
    first_sample = next(loader.iter_samples(args.split, val_ratio=args.val_ratio, seed=args.seed, strict=args.strict))

    backend = LayoutLMv3TokenClassifier(checkpoint)
    prediction = backend.predict(
        first_sample.image_path,
        first_sample.ocr_lines,
        first_sample.ocr_tokens,
        first_sample.image_width,
        first_sample.image_height,
    )
    structured = prediction.result.to_dict() if prediction.result is not None else {}

    artifact: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "compatibility": {
            "is_compatible": compatibility.is_compatible,
            "is_legacy": compatibility.is_legacy,
            "missing_entities": compatibility.missing_entities,
            "message": compatibility.message,
        },
        "config_mapping_validation": config_payload,
        "inference_smoke": {
            "sample_id": first_sample.sample_id,
            "token_count": len(first_sample.ocr_tokens),
            "predicted_token_labels": len(prediction.token_labels),
            "warnings": prediction.warnings,
            "structured_output_keys": sorted(structured.keys()),
            "has_structured_output": bool(structured),
        },
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("Checkpoint validation complete")
    print(f"Checkpoint: {checkpoint}")
    print(f"Compatibility: {compatibility.message}")
    print(f"Inference sample: {first_sample.sample_id}")
    print(f"Saved artifact: {output_path}")


def _validate_config_mappings(checkpoint: Path) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(str(checkpoint))
    id2label_raw = dict(getattr(config, "id2label", {}) or {})
    label2id_raw = dict(getattr(config, "label2id", {}) or {})
    id2label = {int(k): str(v) for k, v in id2label_raw.items()}
    label2id = {str(k): int(v) for k, v in label2id_raw.items()}

    mismatches: list[str] = []
    for idx, label in id2label.items():
        mapped = label2id.get(label)
        if mapped != idx:
            mismatches.append(f"id2label[{idx}]={label} but label2id[{label}]={mapped}")

    return {
        "num_labels": int(getattr(config, "num_labels", len(id2label))),
        "id2label_size": len(id2label),
        "label2id_size": len(label2id),
        "is_consistent": len(mismatches) == 0,
        "mismatches": mismatches,
    }


if __name__ == "__main__":
    main()
