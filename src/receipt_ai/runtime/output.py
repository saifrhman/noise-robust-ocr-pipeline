from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.receipt_ai.schemas import ReceiptExtractionResult


SCHEMA_KEYS = ("vendor", "invoice", "items", "totals", "payment", "metadata")


def format_result_output(
    result: ReceiptExtractionResult | dict[str, Any],
    *,
    output_mode: str = "full",
    include_confidence: bool = True,
    include_provenance: bool = True,
) -> dict[str, Any]:
    """Return schema-stable JSON without internal/debug-only fields."""
    payload = result.to_dict() if isinstance(result, ReceiptExtractionResult) else deepcopy(result)
    out = {key: deepcopy(payload.get(key, {} if key != "items" else [])) for key in SCHEMA_KEYS}

    metadata = dict(out.get("metadata") or {})
    sanitized_metadata: dict[str, Any] = {
        "mode": metadata.get("mode", ""),
        "source_image": metadata.get("source_image", ""),
        "warnings": list(metadata.get("warnings") or []),
    }

    if output_mode == "full":
        if include_confidence:
            sanitized_metadata["confidence"] = float(metadata.get("confidence") or 0.0)
        if include_provenance:
            sanitized_metadata["field_confidences"] = dict(metadata.get("field_confidences") or {})
            sanitized_metadata["field_provenance"] = dict(metadata.get("field_provenance") or {})

    out["metadata"] = sanitized_metadata
    return out
