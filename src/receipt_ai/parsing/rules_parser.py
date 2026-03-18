from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from src.receipt_ai.parsing.normalization import (
    clean_ocr_text,
    extract_amounts,
    extract_currency,
    extract_date_time,
    normalize_date,
    unique_preserve_order,
)
from src.receipt_ai.schemas import (
    ExtractionMetadata,
    InvoiceInfo,
    OCRLine,
    PaymentInfo,
    ReceiptExtractionResult,
    ReceiptItem,
    TotalsInfo,
    VendorInfo,
)


@dataclass(slots=True)
class ParsedItemDetail:
    qty: float
    line_total: float
    tax_code: str


class RuleBasedReceiptParser:
    """Rule-based parser that converts OCR lines into `ReceiptExtractionResult`."""

    def parse(
        self,
        lines: list[OCRLine],
        *,
        source_image: str | Path,
        confidence: float = 0.0,
        mode: str = "easyocr_rules",
    ) -> ReceiptExtractionResult:
        line_texts = [clean_ocr_text(line.text) for line in lines if line.text and line.text.strip()]
        line_texts = unique_preserve_order(line_texts)

        vendor = self._extract_vendor(line_texts)
        invoice = self._extract_invoice(line_texts)
        items = self._extract_items(line_texts)
        totals = self._extract_totals(line_texts)
        payment = self._extract_payment(line_texts)

        # Fill subtotal from item sums when missing.
        if totals.subtotal <= 0 and items:
            subtotal_guess = sum(max(item.line_total, 0.0) for item in items)
            if subtotal_guess > 0:
                totals.subtotal = round(subtotal_guess, 2)

        # Fill total from payable markers or subtotal+tax fallback.
        if totals.total <= 0:
            if totals.subtotal > 0 and totals.tax >= 0:
                totals.total = round(totals.subtotal + totals.tax + totals.service_charge + totals.rounding, 2)

        totals.currency = totals.currency or extract_currency(line_texts)

        return ReceiptExtractionResult(
            vendor=vendor,
            invoice=invoice,
            items=items,
            totals=totals,
            payment=payment,
            metadata=ExtractionMetadata(
                mode=mode,
                confidence=max(0.0, min(1.0, float(confidence))),
                source_image=Path(source_image).name,
            ),
        )

    def _extract_vendor(self, lines: list[str]) -> VendorInfo:
        vendor_name = ""
        registration_number = ""
        address_lines: list[str] = []

        reg_re = re.compile(r"\(?\d{3,8}-?[A-Z]?\)?")

        header_stop_markers = ["TAX INVOICE", "INVOICE", "BILL", "ORDER", "DATE", "QTY", "TABLE"]

        for i, line in enumerate(lines[:20]):
            up = line.upper()
            # Failure-driven fix: avoid noisy header/meta lines as vendor candidates.
            if not vendor_name and self._looks_like_vendor_candidate(line):
                if not any(marker in up for marker in header_stop_markers):
                    vendor_name = line

            if not registration_number:
                m = reg_re.search(line)
                if m:
                    token = m.group(0).strip("() ")
                    if any(ch.isdigit() for ch in token):
                        registration_number = token

            if vendor_name:
                # capture address until invoice metadata starts
                if i > 0 and vendor_name in lines[: i + 1]:
                    if any(marker in up for marker in ["TAX INVOICE", "INVOICE", "BILL", "DATE", "QTY", "TABLE"]):
                        continue
                    if any(ch.isalpha() for ch in line) and line != vendor_name:
                        if any(marker in up for marker in ["TEL", "FAX", "ROC", "GST", "SST", "REG"]):
                            continue
                        address_lines.append(line)

        # fallback address search around explicit address-like content
        if not address_lines:
            for line in lines[:20]:
                up = line.upper()
                if any(marker in up for marker in ["JALAN", "TAMAN", "SEKSYEN", "SELANGOR", "KUALA", "NO."]):
                    address_lines.append(line)

        if address_lines:
            address_lines = unique_preserve_order(address_lines)

        return VendorInfo(
            name=vendor_name,
            registration_number=registration_number,
            address=", ".join(address_lines[:5]).strip(", "),
        )

    def _extract_invoice(self, lines: list[str]) -> InvoiceInfo:
        info = InvoiceInfo()

        # Failure-driven fix: broaden reference number patterns (NO/NO./#, OCR punctuation).
        bill_re = re.compile(r"\b(?:BILL|BIL|RECEIPT)\s*(?:NO|NO\.|#|NUM|NUMBER)?\s*[:=-]?\s*([A-Z0-9][A-Z0-9-]{1,})", re.IGNORECASE)
        order_re = re.compile(r"\b(?:ORDER|ORD)\s*(?:NO|NO\.|#|NUM|NUMBER)?\s*[:=-]?\s*([A-Z0-9][A-Z0-9-]{1,})", re.IGNORECASE)
        table_re = re.compile(r"\b(?:TABLE|TBL)\s*(?:NO|NO\.|#|NUM|NUMBER)?\s*[:=-]?\s*([A-Z0-9][A-Z0-9-]{0,})", re.IGNORECASE)
        cashier_re = re.compile(r"\bCASHIER\s*[:=-]?\s*(.+)$", re.IGNORECASE)

        for line in lines:
            up = line.upper()

            if not info.invoice_type and "INVOICE" in up:
                info.invoice_type = "Tax Invoice" if "TAX" in up else "Invoice"

            if not info.bill_number:
                m = bill_re.search(line)
                if m:
                    info.bill_number = m.group(1).strip()
                elif "BILL" in up:
                    info.bill_number = self._fallback_reference_token(line)

            if not info.order_number:
                m = order_re.search(line)
                if m:
                    info.order_number = m.group(1).strip()
                elif "ORDER" in up:
                    info.order_number = self._fallback_reference_token(line)

            if not info.table_number:
                m = table_re.search(line)
                if m:
                    info.table_number = m.group(1).strip()
                elif "TABLE" in up:
                    info.table_number = self._fallback_reference_token(line)

            if not info.cashier:
                m = cashier_re.search(line)
                if m:
                    info.cashier = m.group(1).strip()

            # Failure-driven fix: recover date/time from same or neighboring lines.
            if not info.date or not info.time:
                d, t = extract_date_time(line)
                if d and not info.date:
                    info.date = d
                if t and not info.time:
                    info.time = t
                if (not info.date or not info.time) and line:
                    parts = [part.strip() for part in re.split(r"\s{2,}|\|", line) if part.strip()]
                    for part in parts:
                        d2, t2 = extract_date_time(part)
                        if d2 and not info.date:
                            info.date = d2
                        if t2 and not info.time:
                            info.time = t2

        # Neighbor fallback for date/time labels split across lines.
        if not info.date or not info.time:
            for idx, line in enumerate(lines[:-1]):
                up = line.upper()
                nxt = lines[idx + 1]
                if any(k in up for k in ["DATE", "TIME"]) and nxt:
                    d, t = extract_date_time(nxt)
                    if d and not info.date:
                        info.date = d
                    if t and not info.time:
                        info.time = t

        # Additional fallback date parse if still missing.
        if not info.date:
            for line in lines:
                d = normalize_date(line)
                if d:
                    info.date = d
                    break

        return info

    def _extract_items(self, lines: list[str]) -> list[ReceiptItem]:
        header_idx = self._find_item_table_header(lines)
        if header_idx < 0:
            return []

        end_idx = self._find_totals_start(lines, start=header_idx + 1)
        if end_idx <= header_idx:
            return []

        segment = lines[header_idx + 1 : end_idx]

        items: list[ReceiptItem] = []
        pending_detail: ParsedItemDetail | None = None
        pending_name_chunks: list[str] = []

        for row in segment:
            line = row.strip()
            if not line:
                continue

            detail = self._parse_item_detail(line)
            if detail is not None:
                pending_detail = detail
                continue

            if self._looks_like_non_item(line):
                continue

            # Failure-driven fix: robust multi-line item name grouping.
            pending_name_chunks.append(line)

            if pending_detail is not None:
                name = " ".join(pending_name_chunks).strip()
                qty = pending_detail.qty if pending_detail.qty > 0 else 1.0
                line_total = pending_detail.line_total if pending_detail.line_total > 0 else 0.0
                unit_price = round(line_total / qty, 4) if qty > 0 and line_total > 0 else 0.0
                items.append(
                    ReceiptItem(
                        name=name,
                        quantity=qty,
                        unit_price=unit_price,
                        line_total=line_total,
                    )
                )
                pending_detail = None
                pending_name_chunks = []

        # Fallback for patterns like:
        #   <item name>
        #   <tax code>
        #   <amount>
        # or
        #   <item name>
        #   <amount>
        i = 0
        while i < len(segment):
            line = segment[i].strip()
            if not line or self._looks_like_non_item(line):
                i += 1
                continue

            if any(ch.isalpha() for ch in line) and not extract_amounts(line):
                next_line = segment[i + 1].strip() if i + 1 < len(segment) else ""
                next2_line = segment[i + 2].strip() if i + 2 < len(segment) else ""

                # pattern: name -> tax -> amount
                if re.search(r"\b(SR|ZR|TX|GST|SST)\b", next_line.upper()) and extract_amounts(next2_line):
                    amount = extract_amounts(next2_line)[-1]
                    items.append(
                        ReceiptItem(
                            name=line,
                            quantity=1.0,
                            unit_price=round(amount, 4),
                            line_total=round(amount, 2),
                        )
                    )
                    i += 3
                    continue

                # pattern: name -> amount
                if extract_amounts(next_line):
                    amount = extract_amounts(next_line)[-1]
                    items.append(
                        ReceiptItem(
                            name=line,
                            quantity=1.0,
                            unit_price=round(amount, 4),
                            line_total=round(amount, 2),
                        )
                    )
                    i += 2
                    continue

                # pattern: name -> qty/price/detail on next line
                detail = self._parse_item_detail(next_line)
                if detail is not None and detail.line_total > 0:
                    qty = detail.qty if detail.qty > 0 else 1.0
                    line_total = detail.line_total
                    unit_price = round(line_total / qty, 4) if qty > 0 else 0.0
                    items.append(
                        ReceiptItem(
                            name=line,
                            quantity=qty,
                            unit_price=unit_price,
                            line_total=round(line_total, 2),
                        )
                    )
                    i += 2
                    continue

            i += 1

        # Description-first fallback pattern.
        if pending_name_chunks:
            name = " ".join(pending_name_chunks).strip()
            if name:
                items.append(ReceiptItem(name=name, quantity=1.0, unit_price=0.0, line_total=0.0))

        # Remove duplicate OCR artifacts.
        deduped: list[ReceiptItem] = []
        seen: set[tuple[str, float]] = set()
        for item in items:
            key = (re.sub(r"\s+", " ", item.name.upper()).strip(), round(item.line_total, 2))
            if key in seen:
                continue
            seen.add(key)
            if item.line_total > 5000:
                continue
            if item.name and not self._looks_like_non_item(item.name):
                deduped.append(item)

        return deduped

    @staticmethod
    def _find_item_table_header(lines: list[str]) -> int:
        for i, line in enumerate(lines):
            up = line.upper()
            has_qty = "QTY" in up or "OTY" in up
            if has_qty and "DESC" in up:
                return i
            # Fallback noisy header variants missing explicit description token.
            if has_qty and ("TAX" in up or "RM" in up):
                return i
        return -1

    @staticmethod
    def _find_totals_start(lines: list[str], start: int) -> int:
        for i in range(start, len(lines)):
            up = lines[i].upper()
            if any(marker in up for marker in ["TOTAL", "SUBTOTAL", "PAID", "CASH", "GST SUMMARY", "CHANGE"]):
                # Do not stop at item table header containing TOTAL text.
                if ("QTY" in up or "OTY" in up) and "DESC" in up:
                    continue
                return i
        return len(lines)

    @staticmethod
    def _parse_item_detail(line: str) -> ParsedItemDetail | None:
        up = line.upper()
        if any(marker in up for marker in ["TABLE", "BILL", "ORDER", "DATE", "CASHIER"]):
            return None

        qty_match = re.match(r"^\s*(\d{1,3}(?:[.,]\d{1,2})?)\b", line)
        amounts = extract_amounts(line)
        tax_match = re.search(r"\b(SR|ZR|TX|GST|SST)\b", up)

        if not qty_match and not amounts:
            return None

        qty = float(qty_match.group(1).replace(",", ".")) if qty_match else 1.0
        line_total = amounts[-1] if amounts else 0.0

        alpha_count = sum(ch.isalpha() for ch in line)
        if not qty_match and alpha_count >= 6 and not tax_match:
            return None

        if line_total > 5000:
            return None

        return ParsedItemDetail(
            qty=qty if qty > 0 else 1.0,
            line_total=line_total,
            tax_code=tax_match.group(1) if tax_match else "",
        )

    @staticmethod
    def _looks_like_non_item(line: str) -> bool:
        up = line.upper()
        stripped = re.sub(r"\s+", " ", up).strip()
        if stripped in {"RM", "SR", "ZR", "TX", "GST", "SST", "TAX"}:
            return True
        markers = [
            "TAX INVOICE",
            "INVOICE",
            "TABLE",
            "BILL",
            "ORDER",
            "CASHIER",
            "SERVER",
            "PAX",
            "GST SUMMARY",
            "TOTAL",
            "PAID",
            "CHANGE",
            "ROUNDING",
        ]
        return any(marker in up for marker in markers)

    def _extract_totals(self, lines: list[str]) -> TotalsInfo:
        subtotal = 0.0
        service_charge = 0.0
        tax = 0.0
        rounding = 0.0
        total = 0.0

        subtotal_candidates: list[float] = []
        tax_candidates: list[float] = []
        total_candidates: list[float] = []

        for idx, line in enumerate(lines):
            up = line.upper()
            amounts = extract_amounts(line)
            if not amounts:
                continue

            value = amounts[-1]

            # Avoid reading percentage rates such as "@ 6" as total values.
            if "@" in up and value <= 10 and idx + 1 < len(lines):
                nxt_amounts = extract_amounts(lines[idx + 1])
                if nxt_amounts:
                    value = nxt_amounts[-1]

            if "SUBTOTAL" in up or "SUB TOTAL" in up:
                subtotal_candidates.append(value)
                continue

            if "SERVICE" in up and "CHARGE" in up:
                service_charge = max(service_charge, value)
                continue

            if ("TAX" in up or "GST" in up) and "SUMMARY" not in up and "INCL" not in up and "EXCL" not in up:
                tax_candidates.append(value)
                continue

            if "ROUNDING" in up or "ADJUSTMENT" in up:
                rounding = value
                continue

            if "TOTAL" in up or "AMT PAYABLE" in up or "AMOUNT PAYABLE" in up or "NETT" in up:
                # Failure-driven fix: ignore summary rates/quantities misread as totals.
                if value <= 1 and "%" in up:
                    continue
                total_candidates.append(value)

        if subtotal_candidates:
            subtotal = max(subtotal_candidates)
        if tax_candidates:
            tax = max(tax_candidates)
        if total_candidates:
            total = max(total_candidates)

        # Failure-driven fix: keep totals consistent when subtotal/tax are stronger than total line.
        if total <= 0 and subtotal > 0:
            total = subtotal + max(tax, 0.0) + max(service_charge, 0.0) + rounding
        if total > 0 and subtotal <= 0 and tax > 0:
            subtotal = max(0.0, total - tax - service_charge - rounding)

        return TotalsInfo(
            subtotal=round(subtotal, 2),
            service_charge=round(service_charge, 2),
            tax=round(tax, 2),
            rounding=round(rounding, 2),
            total=round(total, 2),
            currency=extract_currency(lines),
        )

    def _extract_payment(self, lines: list[str]) -> PaymentInfo:
        method = ""
        amount_paid = 0.0

        for line in lines:
            up = line.upper()
            amounts = extract_amounts(line)

            if "CASH" in up and "CASHIER" not in up:
                method = method or "cash"
                if amounts:
                    amount_paid = max(amount_paid, amounts[-1])

            if any(k in up for k in ["CARD", "VISA", "MASTERCARD", "DEBIT"]):
                method = method or "card"
                if amounts:
                    amount_paid = max(amount_paid, amounts[-1])

            # Failure-driven fix: e-wallet and transfer style payment mentions.
            if any(k in up for k in ["TNG", "E-WALLET", "EWALLET", "TOUCH N GO", "QR", "DUITNOW", "TRANSFER"]):
                method = method or "ewallet"
                if amounts:
                    amount_paid = max(amount_paid, amounts[-1])

            if "PAID" in up and amounts:
                amount_paid = max(amount_paid, amounts[-1])

        return PaymentInfo(method=method, amount_paid=round(amount_paid, 2))

    @staticmethod
    def _looks_like_vendor_candidate(line: str) -> bool:
        up = line.upper().strip()
        if len(line.strip()) < 4:
            return False
        if sum(ch.isalpha() for ch in line) < 6:
            return False
        if any(marker in up for marker in ["TEL", "FAX", "ROC", "GST", "SST", "REG NO", "RECEIPT NO", "CASHIER"]):
            return False
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", up):
            return False
        return True

    @staticmethod
    def _fallback_reference_token(line: str) -> str:
        # Prefer rightmost alphanumeric chunk after separators.
        parts = re.split(r"[:=#-]", line)
        candidates = [part.strip() for part in parts if part.strip()]
        if candidates:
            tail = candidates[-1]
            m = re.search(r"([A-Z0-9][A-Z0-9-]{1,})", tail, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()
        m = re.search(r"\b([A-Z0-9]{2,}(?:-[A-Z0-9]{1,})?)\b", line, flags=re.IGNORECASE)
        return m.group(1).upper() if m else ""
