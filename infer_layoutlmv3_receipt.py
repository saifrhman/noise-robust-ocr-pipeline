from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import prepare_words_boxes_for_inference
from src.app.extraction_modes import (
    build_easyocr_rules_result,
    build_layoutlmv3_only_result,
    build_hybrid_result,
)
from src.app.receipt_script_parser import parse_receipt_script
from src.layoutlmv3_engine import predict_layoutlmv3_from_words, predict_layoutlmv3_with_hybrid


def to_jsonable(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fine-tuned LayoutLMv3 receipt KIE inference with support for 3 extraction modes."
    )
    parser.add_argument("--image", required=True, help="Path to receipt image")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned LayoutLMv3 checkpoint path")
    parser.add_argument("--ocr-path", default=None, help="Optional SROIE box txt path")
    parser.add_argument(
        "--ocr-mode",
        default="none",
        choices=["none", "clahe", "denoise", "otsu", "adaptive"],
        help="Used when OCR path is not provided and EasyOCR fallback is used.",
    )
    parser.add_argument(
        "--extraction-mode",
        default="layoutlmv3_only",
        choices=["easyocr_rules", "layoutlmv3_only", "hybrid"],
        help="Extraction mode: easyocr_rules (rule-based), layoutlmv3_only (pure model), hybrid (model+parser+fusion)",
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

    model_path = str(checkpoint_path) if looks_like_local else checkpoint_raw
    ocr_text = str(features.get("raw_text", ""))

    # ===========================================================================
    # MODE: EASYOCR + RULES
    # ===========================================================================
    if args.extraction_mode == "easyocr_rules":
        receipt_script = parse_receipt_script(ocr_text) if ocr_text.strip() else {}
        
        payload = build_easyocr_rules_result(
            filename=image_path.name,
            mode_selected=args.ocr_mode,
            chosen_mode=args.ocr_mode,
            mean_conf=features.get("ocr_conf", 0.0),
            score=features.get("ocr_score", 0.0),
            edited_text=ocr_text,
            raw_text=ocr_text,
            ocr_results=features.get("ocr_results", []),
            receipt_script=receipt_script,
        )

    # ===========================================================================
    # MODE: LAYOUTLMV3 ONLY
    # ===========================================================================
    elif args.extraction_mode == "layoutlmv3_only":
        prediction = predict_layoutlmv3_from_words(
            image_rgb=features["image_rgb"],
            words=features["words"],
            boxes=features["boxes"],
            abs_boxes=features["abs_boxes"],
            model_name_or_path=model_path,
            was_truncated=bool(features.get("truncated", False)),
        )

        payload = build_layoutlmv3_only_result(
            filename=image_path.name,
            model_path=model_path,
            mode_selected=args.ocr_mode,
            chosen_mode=args.ocr_mode,
            prediction=prediction,
        )

    # ===========================================================================
    # MODE: HYBRID
    # ===========================================================================
    elif args.extraction_mode == "hybrid":
        # Get LayoutLMv3 predictions
        model_pred, _ = predict_layoutlmv3_with_hybrid(
            image_rgb=features["image_rgb"],
            ocr_results=features.get("ocr_results", []),
            model_name_or_path=model_path,
        )

        # Parse receipt script
        parser_output = parse_receipt_script(ocr_text) if ocr_text.strip() else {}

        # Build hybrid result
        payload = build_hybrid_result(
            filename=image_path.name,
            model_path=model_path,
            mode_selected=args.ocr_mode,
            chosen_mode=args.ocr_mode,
            model_prediction=model_pred,
            parser_output=parser_output,
            ocr_text=ocr_text,
        )

    else:
        raise ValueError(f"Unknown extraction mode: {args.extraction_mode}")

    if args.pretty:
        print(json.dumps(to_jsonable(payload), indent=2))
    else:
        print(json.dumps(to_jsonable(payload)))


if __name__ == "__main__":
    main()
