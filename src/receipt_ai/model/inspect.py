from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.model.alignment import build_weak_bio_labels
from src.receipt_ai.model.inference import LayoutLMv3TokenClassifier
from src.receipt_ai.model.labels import sanitize_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect token-level LayoutLMv3 predictions for one receipt sample.")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--checkpoint", default="outputs/layoutlmv3_sroie")
    parser.add_argument("--output-json", default="outputs/layoutlmv3_inspect.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()

    loader = SROIEDatasetLoader(cfg.paths.data_root)
    sample = loader.load_sample(args.sample_id, args.split)
    weak = build_weak_bio_labels(sample)
    gold_labels = sanitize_labels(weak.labels)

    backend = LayoutLMv3TokenClassifier(args.checkpoint)
    prediction = backend.predict(
        sample.image_path,
        sample.ocr_lines,
        sample.ocr_tokens,
        sample.image_width,
        sample.image_height,
    )

    aligned_length = min(len(sample.ocr_tokens), len(gold_labels), len(prediction.token_labels))
    tokens = []
    for idx in range(aligned_length):
        token = sample.ocr_tokens[idx]
        tokens.append(
            {
                "index": idx,
                "text": token.text,
                "line_id": token.line_id,
                "gold_label": gold_labels[idx],
                "predicted_label": prediction.token_labels[idx],
                "score": float(prediction.token_scores[idx]) if idx < len(prediction.token_scores) else 0.0,
                "bbox": token.bbox.to_xyxy(),
            }
        )

    payload = {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "warnings": prediction.warnings,
        "assumptions": weak.assumptions,
        "structured_prediction": prediction.result.to_dict() if prediction.result is not None else {},
        "ground_truth": sample.ground_truth.to_dict() if sample.ground_truth is not None else None,
        "tokens": tokens,
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved inspection artifact to {output_path}")


if __name__ == "__main__":
    main()