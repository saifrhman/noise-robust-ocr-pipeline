from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
from typing import Any

from src.receipt_ai.config import ReceiptAIConfig


DEFAULT_POLICY_PATH = Path("outputs/runtime/recommended_runtime_config.json")


@dataclass(slots=True)
class RuntimePolicy:
    default_mode: str = "hybrid"
    preferred_checkpoint: str = ""
    fallback_mode_on_model_failure: str = "easyocr_rules"
    allow_legacy_checkpoint: bool = True
    hybrid_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "model_field_confidence": 0.65,
            "model_semantic_strong_confidence": 0.75,
            "model_takeover_margin": 0.12,
            "low_confidence_guard": 0.50,
            "total_item_consistency_tolerance": 0.08,
        }
    )
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_runtime_policy(policy_path: str | Path | None = None) -> RuntimePolicy:
    path = Path(policy_path or DEFAULT_POLICY_PATH).expanduser().resolve()
    if not path.exists():
        return RuntimePolicy()

    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = RuntimePolicy(
        default_mode=str(payload.get("default_mode", "hybrid")),
        preferred_checkpoint=str(payload.get("preferred_checkpoint", "")),
        fallback_mode_on_model_failure=str(payload.get("fallback_mode_on_model_failure", "easyocr_rules")),
        allow_legacy_checkpoint=bool(payload.get("allow_legacy_checkpoint", True)),
        hybrid_thresholds=dict(payload.get("hybrid_thresholds", {})) or RuntimePolicy().hybrid_thresholds,
        evidence=dict(payload.get("evidence", {})),
    )
    return policy


def save_runtime_policy(policy: RuntimePolicy, policy_path: str | Path | None = None) -> Path:
    path = Path(policy_path or DEFAULT_POLICY_PATH).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), indent=2), encoding="utf-8")
    return path


def apply_runtime_policy(cfg: ReceiptAIConfig, policy: RuntimePolicy) -> ReceiptAIConfig:
    if policy.preferred_checkpoint:
        cfg.paths.model_checkpoint = Path(policy.preferred_checkpoint).expanduser().resolve()

    thresholds = policy.hybrid_thresholds
    cfg.thresholds.model_field_confidence = float(thresholds.get("model_field_confidence", cfg.thresholds.model_field_confidence))
    cfg.thresholds.model_semantic_strong_confidence = float(
        thresholds.get("model_semantic_strong_confidence", cfg.thresholds.model_semantic_strong_confidence)
    )
    cfg.thresholds.model_takeover_margin = float(thresholds.get("model_takeover_margin", cfg.thresholds.model_takeover_margin))
    cfg.thresholds.low_confidence_guard = float(thresholds.get("low_confidence_guard", cfg.thresholds.low_confidence_guard))
    cfg.thresholds.total_item_consistency_tolerance = float(
        thresholds.get("total_item_consistency_tolerance", cfg.thresholds.total_item_consistency_tolerance)
    )
    return cfg
