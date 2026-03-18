from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import os
from typing import Any

from src.receipt_ai.config import ReceiptAIConfig


DEFAULT_POLICY_PATH = Path("default_config.json")
LEGACY_POLICY_PATH = Path("outputs/runtime/recommended_runtime_config.json")


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
    output: dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "full",
            "include_confidence": True,
            "include_provenance": True,
        }
    )
    deterministic_seed: int = 42
    decision: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_runtime_policy(policy_path: str | Path | None = None) -> RuntimePolicy:
    path = _resolve_policy_path(policy_path)
    if path is None:
        return RuntimePolicy()
    payload = json.loads(path.read_text(encoding="utf-8"))
    default_policy = RuntimePolicy()
    policy = RuntimePolicy(
        default_mode=str(payload.get("default_mode", default_policy.default_mode)),
        preferred_checkpoint=str(payload.get("preferred_checkpoint", "")),
        fallback_mode_on_model_failure=str(
            payload.get("fallback_mode_on_model_failure", default_policy.fallback_mode_on_model_failure)
        ),
        allow_legacy_checkpoint=bool(payload.get("allow_legacy_checkpoint", default_policy.allow_legacy_checkpoint)),
        hybrid_thresholds=dict(payload.get("hybrid_thresholds", {})) or default_policy.hybrid_thresholds,
        output=dict(payload.get("output", {})) or default_policy.output,
        deterministic_seed=int(payload.get("deterministic_seed", default_policy.deterministic_seed)),
        decision=dict(payload.get("decision", {})),
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


def _resolve_policy_path(policy_path: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if policy_path:
        candidates.append(Path(policy_path))

    env_path = str(os.getenv("RECEIPT_DEFAULT_CONFIG", "")).strip()
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([DEFAULT_POLICY_PATH, LEGACY_POLICY_PATH])

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
    return None
