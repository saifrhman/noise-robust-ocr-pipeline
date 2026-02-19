from __future__ import annotations

from pathlib import Path
from collections import Counter
import re
import argparse


WORD_RE = re.compile(r"[A-Za-z]{3,}")  # words length >= 3


def read_sroie_box_text(box_file: Path) -> str:
    """
    SROIE 'box' format per line:
      x1,y1,x2,y2,x3,y3,x4,y4,transcription
    transcription may contain commas -> take parts[8:]
    """
    lines = box_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    texts = []
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
    return "\n".join(texts)


def tokenize(text: str) -> list[str]:
    # Lowercase and extract word-like tokens
    return [w.lower() for w in WORD_RE.findall(text)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--box_dir",
        type=str,
        default="data/sroie_v2/train/box",
        help="Path to SROIE v2 train/box directory",
    )
    p.add_argument(
        "--out",
        type=str,
        default="assets/lexicon_dictionary.txt",
        help="Output dictionary file for SymSpell (term frequency per line)",
    )
    p.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="Minimum frequency to keep a token",
    )
    p.add_argument(
        "--max_words",
        type=int,
        default=50000,
        help="Max number of tokens to write (most common first)",
    )
    args = p.parse_args()

    box_dir = Path(args.box_dir)
    if not box_dir.exists():
        raise FileNotFoundError(f"box_dir not found: {box_dir}")

    files = sorted(box_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {box_dir}")

    counts = Counter()

    for i, f in enumerate(files, start=1):
        txt = read_sroie_box_text(f)
        counts.update(tokenize(txt))
        if i % 200 == 0:
            print(f"Processed {i}/{len(files)} box files...")

    # Filter + limit
    items = [(w, c) for w, c in counts.items() if c >= args.min_freq]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[: args.max_words]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # SymSpell dictionary format: "term frequency" per line
    with out_path.open("w", encoding="utf-8") as f:
        for w, c in items:
            f.write(f"{w} {c}\n")

    total_unique = len(counts)
    kept_unique = len(items)
    print("\nDone.")
    print(f"Unique tokens found: {total_unique}")
    print(f"Tokens kept (freq >= {args.min_freq}, capped to {args.max_words}): {kept_unique}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
