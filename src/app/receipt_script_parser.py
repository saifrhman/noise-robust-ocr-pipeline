from __future__ import annotations

import re
from typing import Any


MONEY_RE = re.compile(r"\b\d{1,6}[.,]\d{2}\b")
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
DATE_COMPACT_RE = re.compile(r"\b(\d{2})[./-](\d{2})(\d{4})\b")
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d(?:[:.][0-5]\d)?\b")
GST_ID_RE = re.compile(r"GST\s*ID\s*[:#-]?\s*([A-Z0-9-]{6,})", re.IGNORECASE)
COMPANY_REG_RE = re.compile(r"\(?\d{3,8}-?[A-Z]?\)?")
INVOICE_NO_RE = re.compile(r"(?:INVOICE\s*(?:NO|#)?|NO)\s*[:#-]?\s*([A-Z]{0,4}\s*[-:]?\s*\d{3,})", re.IGNORECASE)


def _clean_line(line: str) -> str:
    value = (line or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _norm_amount(value: str) -> str:
    return value.replace(",", ".")


def _extract_amounts(line: str) -> list[str]:
    raw_line = line or ""
    raw_line = re.sub(r"(?<=\d)[oO](?=\d)", "0", raw_line)
    raw_line = re.sub(r"(?<=\d)[oO]\b", "0", raw_line)
    raw_line = raw_line.replace(",", ".")

    out = [_norm_amount(x) for x in MONEY_RE.findall(raw_line)]

    split_amounts = re.findall(r"\b(\d{1,4})\s+(\d{2})\b", raw_line)
    for left, right in split_amounts:
        out.append(f"{int(left)}.{right}")

    if out:
        dedup: list[str] = []
        for value in out:
            if value not in dedup:
                dedup.append(value)
        return dedup

    # OCR sometimes drops decimal points on money values (example: 1200 -> 12.00)
    only_digits = re.fullmatch(r"\s*\d{3,6}\s*", raw_line)
    if only_digits:
        raw = re.sub(r"\D", "", raw_line)
        if len(raw) >= 3:
            return [f"{int(raw[:-2])}.{raw[-2:]}"]
    return []


def _fold_ocr_letters(value: str) -> str:
    # Normalize common OCR confusions for keyword matching.
    out = (value or "").upper()
    out = out.replace("0", "O").replace("1", "I").replace("5", "S")
    out = re.sub(r"[^A-Z]", "", out)
    return out


def _contains_marker(line: str, marker: str) -> bool:
    return _fold_ocr_letters(marker) in _fold_ocr_letters(line)


def _line_has_keyword(line_upper: str, keywords: list[str]) -> bool:
    return any(_contains_marker(line_upper, key) for key in keywords)


def _extract_key_amount(lines: list[str], markers: list[str]) -> str:
    markers_upper = [m.upper() for m in markers]
    for idx, line in enumerate(lines):
        up = line.upper()
        if not _line_has_keyword(up, markers_upper):
            continue

        amounts = _extract_amounts(line)
        if amounts:
            return amounts[-1]

        for j in range(idx + 1, min(idx + 4, len(lines))):
            nxt = _extract_amounts(lines[j])
            if nxt:
                return nxt[-1]

    return ""


def _extract_date(lines: list[str]) -> str:
    for line in lines:
        m1 = DATE_RE.search(line)
        if m1:
            return m1.group(0).replace(".", "/").replace("-", "/")

        m2 = DATE_COMPACT_RE.search(line)
        if m2:
            dd, mm, yyyy = m2.groups()
            return f"{dd}/{mm}/{yyyy}"
    return ""


def _extract_header(lines: list[str]) -> dict[str, Any]:
    merchant = ""
    address_lines: list[str] = []
    gst_id = ""
    company_reg_no = ""
    invoice_no = ""
    date = ""
    time = ""

    skip_words = [
        "TAX INVOICE",
        "INVOICE",
        "TOTAL",
        "QTY",
        "AMOUNT",
        "CHANGE",
        "GST SUMMARY",
    ]

    # Merchant: first strong alphabetic line near top.
    for line in lines[:12]:
        up = line.upper()
        if _line_has_keyword(up, skip_words):
            continue
        if sum(ch.isalpha() for ch in line) >= 6:
            merchant = line
            break

    # Address: lines after merchant until receipt meta starts.
    merchant_idx = lines.index(merchant) if merchant and merchant in lines else -1
    if merchant_idx >= 0:
        for line in lines[merchant_idx + 1 : merchant_idx + 9]:
            up = line.upper()
            if _line_has_keyword(up, ["GST", "TAX INVOICE", "INVOICE", "DATE", "QTY", "TOTAL"]):
                break
            if any(ch.isalpha() for ch in line):
                address_lines.append(line)

    for idx, line in enumerate(lines):
        if not gst_id:
            match = GST_ID_RE.search(line)
            if match:
                gst_id = match.group(1).strip()
            elif _contains_marker(line, "GST ID") and idx + 1 < len(lines):
                next_line = lines[idx + 1]
                token = re.sub(r"\s+", "", next_line)
                if re.fullmatch(r"[A-Z0-9-]{6,}", token, re.IGNORECASE):
                    gst_id = token

        if not company_reg_no:
            match = COMPANY_REG_RE.search(line)
            if match:
                token = match.group(0).strip("() ")
                if token and any(ch.isdigit() for ch in token):
                    company_reg_no = token

        if not invoice_no:
            match = INVOICE_NO_RE.search(line)
            if match:
                invoice_no = _clean_line(match.group(1)).replace(" ", "")

        if not date:
            date = _extract_date([line])

        if not time:
            match = TIME_RE.search(line)
            if match:
                maybe = match.group(0)
                if "." in maybe and len(maybe) <= 5:
                    # Often OCR misreads decimal as time, skip short x.xx style tokens.
                    pass
                else:
                    time = maybe

    return {
        "merchant": merchant,
        "address": ", ".join(address_lines).strip(),
        "gst_id": gst_id,
        "company_reg_no": company_reg_no,
        "invoice_no": invoice_no,
        "date": date,
        "time": time,
    }


def _find_totals_start(lines: list[str]) -> int:
    markers = ["TOTAL", "PAYABLE", "ROUNDING", "PAID AMOUNT", "PAID ARNOUNT", "CHANGE", "GST SUMMARY"]
    for idx, line in enumerate(lines):
        if _line_has_keyword(line.upper(), markers):
            return idx
    return len(lines)


def _extract_line_items(lines: list[str], start_idx: int, end_idx: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if start_idx >= end_idx:
        return out

    skip_markers = [
        "TAX INVOICE",
        "INVOICE",
        "DATE",
        "QTY TAX",
        "OTY TAX",
        "QTY",
        "OTY",
        " RM",
        "GST",
        "TOTAL",
        "PAYABLE",
        "PAID",
        "CHANGE",
        "SUMMARY",
    ]

    idx = start_idx
    while idx < end_idx:
        line = lines[idx]
        up = line.upper()

        if _line_has_keyword(up, skip_markers):
            idx += 1
            continue

        has_letters = any(ch.isalpha() for ch in line)
        if not has_letters:
            idx += 1
            continue

        if re.fullmatch(r"[A-Z]{1,5}\s*[-:]?\s*\d{3,}", line.strip(), re.IGNORECASE):
            idx += 1
            continue

        amounts = _extract_amounts(line)
        qty_match = re.search(r"\b\d{1,3}(?:[.,]\d{1,3})?\b", line)
        tax_match = re.search(r"\b(SR|ZR|TX|GST)\b", up)

        description = line
        qty = qty_match.group(0).replace(",", ".") if qty_match else ""
        tax = tax_match.group(1) if tax_match else ""
        amount = amounts[-1] if amounts else ""

        if amount:
            try:
                if float(amount) > 10000:
                    amount = ""
            except ValueError:
                pass

        # If this looks like a split line item, borrow numeric lines below.
        if (not amount or not tax or not qty) and idx + 1 < end_idx:
            for j in range(idx + 1, min(idx + 4, end_idx)):
                next_line = lines[j]
                next_amounts = _extract_amounts(next_line)
                next_qty_match = re.search(r"\b\d{1,3}(?:[.,]\d{1,3})?\b", next_line)
                next_tax_match = re.search(r"\b(SR|ZR|TX|GST)\b", next_line.upper())

                if next_amounts and not amount:
                    amount = next_amounts[-1]
                if next_qty_match and not qty:
                    qty = next_qty_match.group(0).replace(",", ".")
                if next_tax_match and not tax:
                    tax = next_tax_match.group(1)

                if amount and (qty or tax):
                    idx = j
                    break

        # Keep only likely product lines.
        if len(description) >= 3 and (amount or qty):
            out.append(
                {
                    "item_name": description,
                    "qty": qty,
                    "tax_code": tax,
                    "line_total": amount,
                }
            )

        idx += 1

    return out


def parse_receipt_script(text: str) -> dict[str, Any]:
    lines = [_clean_line(line) for line in (text or "").splitlines() if _clean_line(line)]
    if not lines:
        return {
            "header": {},
            "totals": {},
            "line_items": [],
            "tax_summary": [],
            "script_lines": [],
        }

    header = _extract_header(lines)
    if not header.get("date"):
        header["date"] = _extract_date(lines)

    totals = {
        "subtotal": _extract_key_amount(lines, ["SUBTOTAL", "SUB TOTAL"]),
        "total_excl_gst": _extract_key_amount(lines, ["TOTAL EXCL", "EXCL GST"]),
        "total_incl_gst": _extract_key_amount(lines, ["TOTAL INCL", "INCL GST"]),
        "total_amt_payable": _extract_key_amount(lines, ["TOTAL AMT PAYABLE", "TOTAL AT PAYABLE", "AMT PAYABLE", "TOTAL PAYABLE"]),
        "paid_amount": _extract_key_amount(lines, ["PAID AMOUNT", "PAID ARNOUNT", "CASH"]),
        "change": _extract_key_amount(lines, ["CHANGE"]),
        "rounding_adjustment": _extract_key_amount(lines, ["ROUNDING", "ADJUSTMENT"]),
        "total_qty_tender": _extract_key_amount(lines, ["TOTAL QTY TENDER", "TOTAL OTY TENDER", "QTY TENDER"]),
    }

    # Invoice number can appear as split tokens over multiple lines.
    if not header.get("invoice_no"):
        for idx, line in enumerate(lines[:-1]):
            if line.upper() in {"NO", "NO.", "INVOICE NO", "INVOICE NO."}:
                candidate = re.sub(r"\s+", "", lines[idx + 1])
                if re.search(r"[A-Z0-9-]{4,}", candidate, re.IGNORECASE):
                    header["invoice_no"] = candidate
                    break

    # Tax summary rows: collect lines after GST summary marker.
    tax_summary: list[dict[str, str]] = []
    gst_idx = -1
    for idx, line in enumerate(lines):
        if _contains_marker(line, "GST SUMMARY"):
            gst_idx = idx
            break

    if gst_idx >= 0:
        for line in lines[gst_idx + 1 : gst_idx + 8]:
            up = line.upper()
            if "THANK" in up:
                break
            amounts = _extract_amounts(line)
            if not amounts:
                continue
            code_match = re.search(r"\b(SR|ZR|TX|SST|GST)\b", up)
            tax_summary.append(
                {
                    "code": code_match.group(1) if code_match else "",
                    "raw": line,
                    "amounts": amounts,
                }
            )

    # Items section: between TAX INVOICE header and first totals marker.
    item_start = 0
    for idx, line in enumerate(lines):
        if _contains_marker(line, "TAX INVOICE") or _contains_marker(line, "INVOICE"):
            item_start = idx + 1

    item_end = _find_totals_start(lines)
    line_items = _extract_line_items(lines, start_idx=item_start, end_idx=item_end)

    return {
        "header": header,
        "totals": totals,
        "line_items": line_items,
        "tax_summary": tax_summary,
        "script_lines": lines,
    }
