from __future__ import annotations

from datetime import datetime
import re


_AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,6}(?:[.,]\d{1,2})?)(?!\d)")
_DATE_PATTERNS = [
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%Y-%m-%d",
]


def clean_ocr_text(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""

    replacements = {
        r"\bRRN\b": "RM",
        r"\bN0\.\b": "NO.",
        r"\b0RDER\b": "ORDER",
        r"\bB1LL\b": "BILL",
        r"\b0TY\b": "QTY",
        r"\bARNOUNT\b": "AMOUNT",
        r"\bSST\b": "GST",
    }

    out = value
    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    out = re.sub(r"\s+", " ", out).strip()
    return out


def normalize_amount_text(value: str) -> str:
    out = (value or "").strip()
    out = out.replace(",", ".")
    out = re.sub(r"(?<=\d)[oO](?=\d)", "0", out)
    out = re.sub(r"[^0-9.]", "", out)
    return out


def parse_amount(value: str) -> float:
    text = normalize_amount_text(value)
    if not text:
        return 0.0

    if text.count(".") > 1:
        # Keep last decimal point in very noisy OCR output.
        chunks = text.split(".")
        text = "".join(chunks[:-1]) + "." + chunks[-1]

    try:
        if "." not in text and len(text) >= 3:
            # Example: 1200 may be 12.00 in some OCR contexts.
            return float(f"{int(text[:-2])}.{text[-2:]}")
        return float(text)
    except ValueError:
        return 0.0


def extract_amounts(text: str) -> list[float]:
    amounts: list[float] = []
    for match in _AMOUNT_RE.findall(text or ""):
        parsed = parse_amount(match)
        if parsed > 0:
            amounts.append(parsed)
    return amounts


def extract_currency(lines: list[str]) -> str:
    joined = " ".join(lines).upper()
    if "RM" in joined or "MYR" in joined:
        return "MYR"
    if "$" in joined:
        return "USD"
    if "SGD" in joined:
        return "SGD"
    return ""


def normalize_date(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"\s+", "", text)

    for pattern in _DATE_PATTERNS:
        try:
            dt = datetime.strptime(text, pattern)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    m = re.search(r"\b(\d{2})[/-](\d{2})[/-](\d{4})\b", text)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"

    return ""


def extract_date_time(text: str) -> tuple[str, str]:
    line = text or ""

    date_match = re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", line)
    time_match = re.search(r"\b(?:[01]?\d|2[0-3])[:.]\d{2}(?::\d{2})?\b", line)

    date = normalize_date(date_match.group(0)) if date_match else ""
    time = ""
    if time_match:
        time = time_match.group(0).replace(".", ":")

    return date, time


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", value.strip().upper())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value.strip())
    return out
