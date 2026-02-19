# src/run_sroie_eval.py
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import re

import cv2

from src.preprocess import preprocess_for_ocr, candidate_modes_for_auto
from src.ocr_engine import run_easyocr
from src.evaluate import char_accuracy, normalize_text


def ocr_text_from_results(results: list[dict]) -> str:
    parts = [r.get("text", "") for r in results if r.get("text")]
    return " ".join(parts).strip()


def mean_conf(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(float(r.get("conf", 0.0)) for r in results) / max(len(results), 1)


def text_quality_score(text: str) -> float:
    """
    Cheap sanity signal (no GT):
    - Prefer outputs with lots of alphanumeric content
    - Prefer non-trivially long text (capped)
    """
    t = normalize_text(text)
    if not t:
        return 0.0
    alnum = sum(ch.isalnum() for ch in t)
    length = len(t)
    return (alnum / max(length, 1)) + min(length, 80) / 80.0


def load_gt_text(box_txt_path: Path) -> str:
    """
    SROIE v2 'box' ground truth format per line:
      x1,y1,x2,y2,x3,y3,x4,y4,transcription
    """
    lines = box_txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    texts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 9:
            continue
        transcription = ",".join(parts[8:]).strip()
        if transcription:
            texts.append(transcription)
    return "\n".join(texts).strip()


def run_ocr_on(img_bgr, mode: str) -> dict:
    processed, _ = preprocess_for_ocr(img_bgr, mode=mode)
    results = run_easyocr(processed)
    text = ocr_text_from_results(results)
    conf = mean_conf(results)
    return {
        "mode": mode,
        "processed": processed,
        "results": results,
        "text": text,
        "conf": conf,
    }


def blended_score(conf: float, text: str) -> float:
    # Blend OCR confidence with a light text sanity signal
    return (0.6 * conf) + (0.4 * text_quality_score(text))


def choose_best_auto(img_bgr, margin: float = 0.03) -> dict:
    """
    Conservative auto:
    - Baseline is always 'none'
    - Try other candidates (clahe/denoise by default)
    - Switch only if blended score beats baseline by >= margin
    """
    baseline = run_ocr_on(img_bgr, "none")
    baseline["score"] = blended_score(baseline["conf"], baseline["text"])

    best = baseline

    for mode in candidate_modes_for_auto():
        if mode == "none":
            continue
        out = run_ocr_on(img_bgr, mode)
        out["score"] = blended_score(out["conf"], out["text"])
        if out["score"] > best["score"] + margin:
            best = out

    return best


def eval_split(
    img_dir: Path,
    gt_dir: Path,
    mode: str,
    margin: float,
    max_images: int | None = None,
    save_examples: bool = False,
):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {img_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Missing ground-truth folder: {gt_dir}")

    images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    images.sort()

    if max_images is not None:
        images = images[:max_images]

    rows: list[dict] = []
    improved = 0

    out_dir = Path("outputs") / f"sroie_{img_dir.parent.name}_{mode}"
    if mode == "auto":
        # include margin in folder name to avoid overwriting when you sweep margins
        out_dir = Path("outputs") / f"sroie_{img_dir.parent.name}_{mode}_m{margin:.2f}"
    if save_examples:
        out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(images, start=1):
        gt_path = gt_dir / f"{img_path.stem}.txt"
        if not gt_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_text = load_gt_text(gt_path)

        # Baseline OCR (always none/grayscale)
        base_out = run_ocr_on(img, mode="none")
        base_out["score"] = blended_score(base_out["conf"], base_out["text"])
        base_text = base_out["text"]
        base_acc = char_accuracy(base_text, gt_text)

        # Preproc OCR
        if mode == "auto":
            pre_out = choose_best_auto(img, margin=margin)
            chosen_mode = pre_out["mode"]
        else:
            pre_out = run_ocr_on(img, mode=mode)
            pre_out["score"] = blended_score(pre_out["conf"], pre_out["text"])
            chosen_mode = mode

        pre_text = pre_out["text"]
        pre_acc = char_accuracy(pre_text, gt_text)

        delta = pre_acc - base_acc
        if delta > 0:
            improved += 1

        rows.append(
            {
                "file": img_path.name,
                "mode": mode,
                "chosen_mode": chosen_mode,
                "margin": f"{margin:.2f}" if mode == "auto" else "",
                "base_char_acc": f"{base_acc:.4f}",
                "pre_char_acc": f"{pre_acc:.4f}",
                "delta": f"{delta:.4f}",
                "base_conf": f"{base_out['conf']:.4f}",
                "pre_conf": f"{pre_out['conf']:.4f}",
                "base_score": f"{base_out['score']:.4f}",
                "pre_score": f"{pre_out['score']:.4f}",
                "gt_preview": normalize_text(gt_text)[:160],
                "base_preview": normalize_text(base_text)[:160],
                "pre_preview": normalize_text(pre_text)[:160],
            }
        )

        if save_examples and idx <= 10:
            cv2.imwrite(str(out_dir / f"{img_path.stem}_processed_{chosen_mode}.png"), pre_out["processed"])

        if idx % 50 == 0:
            print(f"[{img_dir.parent.name} | {mode}] processed {idx}/{len(images)}...")

    def mean(vals: list[str]) -> float:
        xs = [float(v) for v in vals]
        return sum(xs) / max(len(xs), 1)

    base_mean = mean([r["base_char_acc"] for r in rows]) if rows else 0.0
    pre_mean = mean([r["pre_char_acc"] for r in rows]) if rows else 0.0

    print(f"\n=== {img_dir.parent.name.upper()} SUMMARY (mode={mode}) ===")
    print(f"Samples evaluated: {len(rows)}")
    print(f"Mean char accuracy (baseline): {base_mean:.4f}")
    print(f"Mean char accuracy (preproc):  {pre_mean:.4f}")
    print(f"Improved cases: {improved}/{len(rows)}")

    return rows, base_mean, pre_mean


def main():
    p = argparse.ArgumentParser(description="Evaluate OCR baseline vs preprocessing on SROIE v2 (Kaggle).")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument(
        "--mode",
        default="auto",
        choices=["none", "denoise", "clahe", "otsu", "adaptive", "auto"],
        help="Preprocessing mode. 'auto' tries multiple and picks best by blended score.",
    )
    p.add_argument("--margin", type=float, default=0.03, help="Auto-mode switch margin (lower = more switching).")
    p.add_argument("--max", type=int, default=200, help="Limit number of images for a quick run.")
    p.add_argument("--save_examples", action="store_true", help="Save first 10 processed images to outputs/.")
    args = p.parse_args()

    root = Path("data") / "sroie_v2" / args.split
    img_dir = root / "img"
    gt_dir = root / "box"

    rows, base_mean, pre_mean = eval_split(
        img_dir=img_dir,
        gt_dir=gt_dir,
        mode=args.mode,
        margin=args.margin,
        max_images=args.max,
        save_examples=args.save_examples,
    )

    out_csv = Path("outputs") / f"sroie_{args.split}_{args.mode}_results.csv"
    if args.mode == "auto":
        out_csv = Path("outputs") / f"sroie_{args.split}_{args.mode}_m{args.margin:.2f}_results.csv"
    out_csv.parent.mkdir(exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else [
        "file",
        "mode",
        "chosen_mode",
        "margin",
        "base_char_acc",
        "pre_char_acc",
        "delta",
        "base_conf",
        "pre_conf",
        "base_score",
        "pre_score",
        "gt_preview",
        "base_preview",
        "pre_preview",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved results CSV: {out_csv}")
    print(
        f"Split: {args.split} | mode={args.mode}"
        + (f" | margin={args.margin:.2f}" if args.mode == "auto" else "")
        + f" | baseline_mean={base_mean:.4f} | preproc_mean={pre_mean:.4f}"
    )


if __name__ == "__main__":
    main()
