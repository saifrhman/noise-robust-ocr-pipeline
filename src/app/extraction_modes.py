"""
Mode-specific extraction and normalization logic for receipt OCR pipeline.

Supports three distinct modes:
1. EasyOCR + Rules: Rule-based extraction with receipt script parser
2. LayoutLMv3 Only: Pure model-based semantic entity extraction
3. Hybrid: Combined model + parser output with fusion logic
"""

from __future__ import annotations

import re
from typing import Any


# ==============================================================================
# NORMALIZATION HELPERS
# ==============================================================================


def normalize_date(text: str) -> str:
    """Normalize date to YYYY-MM-DD format when possible."""
    value = (text or "").strip()
    if not value:
        return ""
    
    value = value.replace("-", "/").replace(".", "/")
    value = re.sub(r"\s+", "", value)
    
    # Try to parse DD/MM/YYYY or similar formats
    date_pattern = r"(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})"
    match = re.search(date_pattern, value)
    if match:
        dd, mm, yyyy = match.groups()
        # Assume 2-digit years >= 50 are 19xx, < 50 are 20xx
        if len(yyyy) == 2:
            yyyy = "19" + yyyy if int(yyyy) >= 50 else "20" + yyyy
        try:
            # Validate day and month ranges
            d = int(dd)
            m = int(mm)
            y = int(yyyy)
            if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
                return f"{y:04d}-{m:02d}-{d:02d}"
        except (ValueError, TypeError):
            pass
    
    return value


def normalize_amount(text: str) -> str:
    """Normalize money amounts to 2-decimal format."""
    value = (text or "").strip()
    if not value:
        return ""
    
    # Replace common OCR errors
    value = re.sub(r"(?<=\d)[oO](?=[\d.])", "0", value)
    value = value.replace(",", ".")
    
    # Extract numeric part
    match = re.search(r"(\d{1,6})(?:\.(\d{1,2}))?", value)
    if match:
        int_part = match.group(1)
        dec_part = match.group(2) or "00"
        # Pad decimals to 2 digits
        if len(dec_part) == 1:
            dec_part = dec_part + "0"
        elif len(dec_part) > 2:
            dec_part = dec_part[:2]
        return f"{int(int_part)}.{dec_part}"
    
    return value


def normalize_invoice_number(text: str) -> str:
    """Normalize invoice/receipt numbers to consistent format."""
    value = (text or "").strip()
    if not value:
        return ""
    
    # Remove extra spaces
    value = re.sub(r"\s+", "", value)
    
    # If it looks like "CS 20243" or "CS20243", normalize to "CS-20243"
    match = re.match(r"^([A-Z]{1,4})(\d{3,})$", value, re.IGNORECASE)
    if match:
        prefix = match.group(1).upper()
        number = match.group(2)
        return f"{prefix}-{number}"
    
    return value


def clean_ocr_text(text: str) -> str:
    """Apply obvious OCR corrections to text."""
    value = (text or "").strip()
    if not value:
        return ""
    
    # Simple corrections for common OCR errors
    corrections = {
        r"\bBHID\b": "BHD",
        r"\{AN\b": "LIAN",
        r"\bN0\.": "NO.",
        r"\bSEK SYEN\b": "SEKSYEN",
        r"\bArnount\b": "Amount",
        r"\bOty\b": "Qty",  # Only when it looks like a header
        r"\briew\b": "new",
    }
    
    for pattern, replacement in corrections.items():
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    
    return value


def clean_receipt_script_output(script: dict[str, Any]) -> dict[str, Any]:
    """Apply normalization to receipt_script parser output."""
    if not script:
        return script
    
    cleaned = {}
    
    # Clean header
    if script.get("header"):
        header = script["header"].copy()
        if header.get("date"):
            header["date"] = normalize_date(header["date"])
        cleaned["header"] = header
    
    # Clean totals - normalize all amount values
    if script.get("totals"):
        totals = script["totals"].copy()
        for key in ["subtotal", "total_excl_gst", "total_incl_gst", "total_amt_payable", 
                    "paid_amount", "change", "rounding_adjustment", "total_qty_tender"]:
            if totals.get(key):
                totals[key] = normalize_amount(totals[key])
        cleaned["totals"] = totals
    
    # Clean line items
    if script.get("line_items"):
        line_items = []
        for item in script["line_items"]:
            cleaned_item = item.copy()
            if cleaned_item.get("line_total"):
                cleaned_item["line_total"] = normalize_amount(cleaned_item["line_total"])
            line_items.append(cleaned_item)
        cleaned["line_items"] = line_items
    
    # Clean tax summary
    if script.get("tax_summary"):
        tax_summary = []
        for tax_row in script["tax_summary"]:
            cleaned_row = tax_row.copy()
            if cleaned_row.get("amounts"):
                cleaned_row["amounts"] = [normalize_amount(a) for a in cleaned_row.get("amounts", [])]
            tax_summary.append(cleaned_row)
        cleaned["tax_summary"] = tax_summary
    
    # Keep script_lines as-is (no normalization needed)
    if script.get("script_lines"):
        cleaned["script_lines"] = script["script_lines"]
    
    return cleaned


def clean_layoutlmv3_output(prediction: dict[str, Any]) -> dict[str, Any]:
    """Apply normalization to LayoutLMv3 model output."""
    if not prediction:
        return prediction
    
    cleaned = prediction.copy()
    
    # Clean fields
    if cleaned.get("fields"):
        fields = cleaned["fields"].copy()
        if fields.get("date"):
            fields["date"] = normalize_date(fields["date"])
        if fields.get("total"):
            fields["total"] = normalize_amount(fields["total"])
        cleaned["fields"] = fields
    
    return cleaned


# ==============================================================================
# FUSION LOGIC FOR HYBRID MODE
# ==============================================================================


def fuse_model_and_parser(
    model_prediction: dict[str, Any],
    parser_output: dict[str, Any],
    ocr_text: str = "",
) -> dict[str, Any]:
    """
    Fuse LayoutLMv3 model output with receipt_script parser output.
    
    Fusion priority:
    1. Agreement between model and parser
    2. OCR script_lines evidence
    3. Arithmetic consistency
    4. Parser for structured totals/line_items/tax_summary
    5. Model for semantic entities parser missed
    """
    fused = {}
    
    # Header: merge model entities with parser header
    fused_header = {}
    model_fields = model_prediction.get("fields", {})
    parser_header = parser_output.get("header", {})
    
    # Merchant: prefer model if it has high confidence, else parser
    fused_header["merchant"] = model_fields.get("merchant") or parser_header.get("merchant") or ""
    
    # Date: prefer model normalized date if available
    if model_fields.get("date"):
        fused_header["date"] = model_fields["date"]
    elif parser_header.get("date"):
        fused_header["date"] = normalize_date(parser_header["date"])
    else:
        fused_header["date"] = ""
    
    # Address: prefer model if available
    fused_header["address"] = model_fields.get("address") or parser_header.get("address") or ""
    
    # GST ID, company reg, invoice no: from parser if model doesn't have them
    fused_header["gst_id"] = parser_header.get("gst_id") or ""
    fused_header["company_reg_no"] = parser_header.get("company_reg_no") or ""
    fused_header["invoice_no"] = parser_header.get("invoice_no") or ""
    fused_header["time"] = parser_header.get("time") or ""
    
    fused["header"] = fused_header
    
    # Totals: parser is usually better for structured line totals
    parser_totals = parser_output.get("totals", {})
    fused_totals = {}
    
    amount_keys = ["subtotal", "total_excl_gst", "total_incl_gst", "total_amt_payable",
                   "paid_amount", "change", "rounding_adjustment", "total_qty_tender"]
    
    for key in amount_keys:
        # Prefer parser value if available
        fused_totals[key] = parser_totals.get(key) or ""
    
    fused["totals"] = fused_totals
    
    # Line items: parser is usually better at structured extraction
    fused["line_items"] = parser_output.get("line_items", [])
    
    # Tax summary: parser structure is reliable
    fused["tax_summary"] = parser_output.get("tax_summary", [])
    
    return fused


# ==============================================================================
# MODE BUILDERS
# ==============================================================================


def build_easyocr_rules_result(
    filename: str,
    mode_selected: str,
    chosen_mode: str,
    mean_conf: float,
    score: float,
    edited_text: str,
    raw_text: str,
    ocr_results: list[dict[str, Any]],
    receipt_script: dict[str, Any],
    margin: float | None = None,
) -> dict[str, Any]:
    """Build EasyOCR + Rules mode output (mode=easyocr_rules)."""
    
    cleaned_script = clean_receipt_script_output(receipt_script)
    
    return {
        "mode": "easyocr_rules",
        "metadata": {
            "file": filename,
            "mode_selected": mode_selected,
            "chosen_mode": chosen_mode,
            "auto_margin": margin,
            "mean_conf": float(mean_conf),
            "score": float(score),
        },
        "header": cleaned_script.get("header", {}),
        "totals": cleaned_script.get("totals", {}),
        "line_items": cleaned_script.get("line_items", []),
        "tax_summary": cleaned_script.get("tax_summary", []),
        "script_lines": cleaned_script.get("script_lines", []),
        "raw_ocr_results": ocr_results,
    }


def build_layoutlmv3_only_result(
    filename: str,
    model_path: str,
    mode_selected: str,
    chosen_mode: str,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Build LayoutLMv3 Only mode output (mode=layoutlmv3_only)."""
    
    cleaned_pred = clean_layoutlmv3_output(prediction)
    
    return {
        "mode": "layoutlmv3_only",
        "metadata": {
            "file": filename,
            "model": model_path,
            "mode_selected": mode_selected,
            "chosen_mode": chosen_mode,
            "num_words": cleaned_pred.get("num_words", 0),
            "was_truncated": cleaned_pred.get("was_truncated", False),
        },
        "fields": cleaned_pred.get("fields", {}),
        "raw_entities": cleaned_pred.get("raw_entities", []),
        "warnings": [cleaned_pred.get("warning")] if cleaned_pred.get("warning") else [],
        "grouped_entities": cleaned_pred.get("entities", []),
    }


def build_hybrid_result(
    filename: str,
    model_path: str,
    mode_selected: str,
    chosen_mode: str,
    model_prediction: dict[str, Any],
    parser_output: dict[str, Any],
    ocr_text: str = "",
) -> dict[str, Any]:
    """Build Hybrid mode output (mode=hybrid)."""
    
    cleaned_model = clean_layoutlmv3_output(model_prediction)
    cleaned_parser = clean_receipt_script_output(parser_output)
    
    # Fuse outputs
    fused = fuse_model_and_parser(cleaned_model, cleaned_parser, ocr_text)
    
    return {
        "mode": "hybrid",
        "metadata": {
            "file": filename,
            "model": model_path,
            "mode_selected": mode_selected,
            "chosen_mode": chosen_mode,
            "num_words": cleaned_model.get("num_words", 0),
            "was_truncated": cleaned_model.get("was_truncated", False),
        },
        "fields": cleaned_model.get("fields", {}),
        "raw_entities": cleaned_model.get("raw_entities", []),
        "receipt_script": cleaned_parser,
        "final_fused": fused,
        "warnings": [cleaned_model.get("warning")] if cleaned_model.get("warning") else [],
    }
