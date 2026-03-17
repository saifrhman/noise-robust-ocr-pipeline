from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import prepare_words_boxes_for_inference
from src.layoutlmv3_engine import predict_layoutlmv3_from_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fine-tuned LayoutLMv3 receipt KIE inference.")
    parser.add_argument("--image", required=True, help="Path to receipt image")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned LayoutLMv3 checkpoint path")
    parser.add_argument("--ocr-path", default=None, help="Optional SROIE box txt path")
    parser.add_argument(
        "--ocr-mode",
        default="none",
        choices=["none", "clahe", "denoise", "otsu", "adaptive"],
        help="Used when OCR path is not provided and EasyOCR fallback is used.",
    )
    parser.add_argument("--max-words", type=int, default=512)
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint_raw = str(args.checkpoint).strip()
    checkpoint_path = Path(checkpoint_raw).expanduser().resolve()

    # If checkpoint looks like a local path, require it to exist so users get a clear message.
    looks_like_local = "/" in checkpoint_raw or checkpoint_raw.startswith(".")
    if looks_like_local and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint folder not found: {checkpoint_path}. "
            "Train first with train_layoutlmv3_receipts.py or provide a valid model path."
        )

    if looks_like_local and checkpoint_path.exists():
        required_any = ["pytorch_model.bin", "model.safetensors"]
        required_all = ["config.json", "preprocessor_config.json"]

        missing_all = [name for name in required_all if not (checkpoint_path / name).exists()]
        has_any_model = any((checkpoint_path / name).exists() for name in required_any)

        if missing_all or not has_any_model:
            raise FileNotFoundError(
                f"Checkpoint directory is incomplete: {checkpoint_path}. "
                "Expected files: config.json, preprocessor_config.json, and model weights "
                "(pytorch_model.bin or model.safetensors). Re-run training or use a complete checkpoint."
            )

    ocr_path = Path(args.ocr_path).resolve() if args.ocr_path else None

    features = prepare_words_boxes_for_inference(
        image_path=image_path,
        ocr_path=ocr_path,
        ocr_mode=args.ocr_mode,
        max_words=args.max_words,
    )

    prediction = predict_layoutlmv3_from_words(
        image_rgb=features["image_rgb"],
        words=features["words"],
        boxes=features["boxes"],
        abs_boxes=features["abs_boxes"],
        model_name_or_path=str(checkpoint_path) if looks_like_local else checkpoint_raw,
        was_truncated=bool(features.get("truncated", False)),
    )

    ocr_text = str(features.get("raw_text", ""))
    if ocr_text.strip():
        from src.app.receipt_script_parser import parse_receipt_script
        prediction["receipt_script"] = parse_receipt_script(ocr_text)

    if prediction.get("warning"):
        # Keep output machine-readable but explicit.
        payload = {
            "merchant": "",
            "date": "",
            "address": "",
            "total": "",
            "raw_entities": [],
            "receipt_script": {},
            "warning": prediction.get("warning", ""),
        }
    else:
        fields = prediction.get("fields", {})
        payload = {
            "merchant": fields.get("merchant", ""),
            "date": fields.get("date", ""),
            "address": fields.get("address", ""),
            "total": fields.get("total", ""),
            "raw_entities": prediction.get("raw_entities", []),
            "receipt_script": prediction.get("receipt_script", {}),
        }

    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
