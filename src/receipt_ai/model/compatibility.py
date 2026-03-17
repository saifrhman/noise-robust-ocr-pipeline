from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .labels import ENTITY_NAMES, normalize_model_label


@dataclass(slots=True)
class CheckpointCompatibility:
    is_compatible: bool
    is_legacy: bool
    normalized_labels: list[str]
    missing_entities: list[str]
    message: str


def inspect_label_mapping(id2label: dict[int, str] | dict[str, str] | None) -> CheckpointCompatibility:
    raw_id2label = {str(k): str(v) for k, v in dict(id2label or {}).items()}
    return _build_compatibility(raw_id2label)


def inspect_checkpoint_label_space(model_name_or_path: str | Path) -> CheckpointCompatibility:
    """Inspect checkpoint label space and detect legacy/incompatible schemas."""
    path = Path(model_name_or_path).expanduser()
    config_path = path / "config.json" if path.exists() else None

    raw_id2label: dict[str, str] = {}
    if config_path and config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw_id2label = {str(k): str(v) for k, v in dict(payload.get("id2label", {})).items()}

    return _build_compatibility(raw_id2label)


def _build_compatibility(raw_id2label: dict[str, str]) -> CheckpointCompatibility:
    normalized_labels = sorted({normalize_model_label(label) for label in raw_id2label.values()})
    normalized_entities = sorted({label[2:] for label in normalized_labels if label.startswith(("B-", "I-"))})
    missing_entities = [entity for entity in ENTITY_NAMES if entity not in normalized_entities]

    is_legacy = normalized_entities != [] and set(normalized_entities).issubset({"VENDOR_NAME", "ADDRESS", "DATE", "TOTAL"})
    is_compatible = len(missing_entities) == 0

    if is_compatible:
        message = "Checkpoint label space matches the richer receipt BIO schema."
    elif is_legacy:
        message = (
            "Checkpoint uses a legacy reduced label space (typically vendor/address/date/total only). "
            "It can run for limited inference but does not support the full richer schema."
        )
    else:
        message = (
            "Checkpoint label space does not match the richer receipt BIO schema. "
            "Retrain with the current label maps for full functionality."
        )

    return CheckpointCompatibility(
        is_compatible=is_compatible,
        is_legacy=is_legacy,
        normalized_labels=normalized_labels,
        missing_entities=missing_entities,
        message=message,
    )
