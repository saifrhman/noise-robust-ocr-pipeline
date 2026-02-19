import argparse
from pathlib import Path
import cv2

from src.preprocess import preprocess_for_ocr
from src.ocr_engine import run_easyocr, best_text
from src.evaluate import char_accuracy

def parse_args():
    p = argparse.ArgumentParser(description="Noise-robust OCR micro-project (preprocess + EasyOCR + eval).")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--gt", default=None, help="Ground truth text (optional)")
    p.add_argument("--save", action="store_true", help="Save processed image to outputs/")
    return p.parse_args()

def main():
    args = parse_args()
    img_path = Path(args.image)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    processed, _ = preprocess_for_ocr(img)
    results = run_easyocr(processed)
    chosen_text, chosen_conf = best_text(results)

    print("=== OCR RESULTS (top 5) ===")
    if results:
        for i, r in enumerate(sorted(results, key=lambda x: x["conf"], reverse=True)[:5], start=1):
            print(f"{i}. conf={r['conf']:.3f} text={r['text']}")
    else:
        print("No text detected.")

    print("\n=== BEST PICK ===")
    print(f"best_conf={chosen_conf:.3f}")
    print(f"best_text={chosen_text}")

    if args.gt is not None:
        score = char_accuracy(chosen_text, args.gt)
        print("\n=== QUICK EVAL ===")
        print(f"GT:   {args.gt}")
        print(f"PRED: {chosen_text}")
        print(f"char_accuracy={score:.3f}")

    if args.save:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{img_path.stem}_processed.png"
        cv2.imwrite(str(out_path), processed)
        print(f"\nSaved processed image: {out_path}")

if __name__ == "__main__":
    main()
