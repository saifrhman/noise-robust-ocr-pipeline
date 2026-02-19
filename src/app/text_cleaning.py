from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from symspellpy import SymSpell, Verbosity


# -------------------------
# Safe normalization (generic)
# -------------------------
def normalize_text_light(text: str) -> str:
    t = text

    # Time: 17:09;21 -> 17:09:21
    t = re.sub(r"(\d{1,2}):(\d{2})[;](\d{2})", r"\1:\2:\3", t)

    # Comma decimals: 38,90 -> 38.90
    t = re.sub(r"(\d)\s*,\s*(\d{2})\b", r"\1.\2", t)

    # Broken decimals: 11,.10 -> 11.10
    t = re.sub(r"(\d)\s*,\.\s*(\d{2})\b", r"\1.\2", t)

    # Normalize possessives: McDonald ' s -> McDonald's (generic)
    t = re.sub(r"\b([A-Za-z]+)\s*'\s*s\b", r"\1's", t)

    # Remove spaces before punctuation
    t = re.sub(r"\s+([,.;:])", r"\1", t)

    # Collapse whitespace
    t = re.sub(r"\s{2,}", " ", t)

    return t.strip()


# -------------------------
# Lexicon correction (data-driven)
# -------------------------
_WORD_RE = re.compile(r"[A-Za-z]{3,}")  # only correct word-like tokens length >= 3

_symspell_singleton: Optional[SymSpell] = None


def _load_symspell(lexicon_dict_path: str | Path, max_edit_distance: int = 2) -> Optional[SymSpell]:
    lex_path = Path(lexicon_dict_path)
    if not lex_path.exists():
        return None

    sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
    ok = sym.load_dictionary(str(lex_path), term_index=0, count_index=1, separator=" ")
    if not ok:
        return None
    return sym


def _preserve_case(original: str, corrected: str) -> str:
    if not corrected:
        return original
    if original.isupper():
        return corrected.upper()
    if original[0].isupper():
        return corrected.capitalize()
    return corrected


def correct_with_lexicon(text: str, sym: SymSpell, max_edit_distance: int = 2) -> str:
    """
    Correct tokens using SymSpell TOP suggestion.
    Keeps numbers/currency untouched by only targeting [A-Za-z]{3,}.
    """
    def repl(match: re.Match) -> str:
        w = match.group(0)

        # --- Guards: don't "spell-correct" IDs/acronyms/brand blocks ---
        if any(ch.isdigit() for ch in w):
            return w
        if w.isupper():              # CROSS, CHANNEL, SDN, BHD
            return w
        if len(w) < 4:
            return w

        suggestions = sym.lookup(w.lower(), Verbosity.TOP, max_edit_distance=max_edit_distance)
        if not suggestions:
            return w

        best = suggestions[0].term
        return _preserve_case(w, best)



def clean_ocr_text(text: str, *args, **kwargs) -> str:
    """
    Clean OCR text but PRESERVE line breaks.
    Applies cleaning per line and rejoins with '\n'.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    for ln in lines:
        ln2 = ln

        # --- keep whatever normalization you already do, but DO NOT use \s to collapse everything ---
        # Example safe normalizations (keep yours if you already have them):
        ln2 = re.sub(r"(\d{1,2}):(\d{2})[;](\d{2})", r"\1:\2:\3", ln2)
        ln2 = re.sub(r"(\d)\s*,\s*(\d{2})\b", r"\1.\2", ln2)
        ln2 = re.sub(r"(\d)\s*,\.\s*(\d{2})\b", r"\1.\2", ln2)
        ln2 = re.sub(r"\s+([,.;:])", r"\1", ln2)

        # Collapse spaces/tabs ONLY (not newlines)
        ln2 = re.sub(r"[ \t]{2,}", " ", ln2).strip()

        cleaned_lines.append(ln2)

    # Rejoin with newlines so extraction works
    out = "\n".join([l for l in cleaned_lines if l != ""])
    return out.strip()
