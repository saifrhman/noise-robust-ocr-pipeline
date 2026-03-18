from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
from typing import Any

from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.runtime.policy import DEFAULT_POLICY_PATH, load_runtime_policy


DEFAULT_REGISTRY_PATH = Path("outputs/runtime/checkpoint_registry.json")


@dataclass(slots=True)
class CheckpointEntry:
    path: str
    schema_status: str
    is_legacy: bool
    missing_entities: list[str]
    compatibility_message: str
    training_summary: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    evaluation_summary: dict[str, Any] = field(default_factory=dict)
    recommended_use: str = "unknown"
    modified_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CheckpointRegistry:
    generated_at_epoch: float
    active_default_checkpoint: str
    last_trained_checkpoint: str
    best_validated_checkpoint: str
    checkpoints: list[CheckpointEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_epoch": self.generated_at_epoch,
            "active_default_checkpoint": self.active_default_checkpoint,
            "last_trained_checkpoint": self.last_trained_checkpoint,
            "best_validated_checkpoint": self.best_validated_checkpoint,
            "checkpoints": [entry.to_dict() for entry in self.checkpoints],
        }


def build_checkpoint_registry(
    checkpoint_paths: list[Path],
    *,
    outputs_root: Path,
) -> CheckpointRegistry:
    entries: list[CheckpointEntry] = []
    for ckpt in checkpoint_paths:
        entry = _build_entry(ckpt, outputs_root)
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda e: e.modified_time, reverse=True)

    policy = load_runtime_policy(DEFAULT_POLICY_PATH)
    active_default = str(Path(policy.preferred_checkpoint).expanduser().resolve()) if policy.preferred_checkpoint else ""

    last_trained = entries[0].path if entries else ""
    best_validated = _select_best_validated(entries)

    import time

    return CheckpointRegistry(
        generated_at_epoch=float(time.time()),
        active_default_checkpoint=active_default,
        last_trained_checkpoint=last_trained,
        best_validated_checkpoint=best_validated,
        checkpoints=entries,
    )


def save_checkpoint_registry(registry: CheckpointRegistry, registry_path: str | Path | None = None) -> Path:
    path = Path(registry_path or DEFAULT_REGISTRY_PATH).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry.to_dict(), indent=2), encoding="utf-8")
    return path


def load_checkpoint_registry(registry_path: str | Path | None = None) -> dict[str, Any] | None:
    path = Path(registry_path or DEFAULT_REGISTRY_PATH).expanduser().resolve()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_checkpoints(outputs_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for config_path in outputs_root.glob("**/config.json"):
        ckpt_dir = config_path.parent
        if (ckpt_dir / "pytorch_model.bin").exists() or (ckpt_dir / "model.safetensors").exists():
            candidates.append(ckpt_dir)

    # Remove nested duplicates.
    uniq = sorted({str(path.resolve()) for path in candidates})
    return [Path(path) for path in uniq]


def _build_entry(ckpt: Path, outputs_root: Path) -> CheckpointEntry | None:
    if not ckpt.exists():
        return None

    compatibility = inspect_checkpoint_label_space(ckpt)
    if compatibility.is_compatible:
        schema_status = "richer_schema"
        recommended_use = "recommended"
    elif compatibility.is_legacy:
        schema_status = "legacy_reduced_schema"
        recommended_use = "limited_legacy"
    else:
        schema_status = "incompatible"
        recommended_use = "not_recommended"

    training_summary = _extract_training_summary(ckpt)
    validation_summary = _extract_validation_summary(outputs_root, ckpt)
    evaluation_summary = _extract_evaluation_summary(outputs_root, ckpt)

    return CheckpointEntry(
        path=str(ckpt.resolve()),
        schema_status=schema_status,
        is_legacy=compatibility.is_legacy,
        missing_entities=compatibility.missing_entities,
        compatibility_message=compatibility.message,
        training_summary=training_summary,
        validation_summary=validation_summary,
        evaluation_summary=evaluation_summary,
        recommended_use=recommended_use,
        modified_time=float(ckpt.stat().st_mtime),
    )


def _extract_training_summary(ckpt: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    trainer_state = ckpt / "trainer_state.json"
    if trainer_state.exists():
        try:
            payload = json.loads(trainer_state.read_text(encoding="utf-8"))
            summary["global_step"] = payload.get("global_step")
            summary["best_metric"] = payload.get("best_metric")
            summary["epoch"] = payload.get("epoch")
            summary["train_runtime"] = payload.get("log_history", [{}])[-1].get("train_runtime") if payload.get("log_history") else None
        except json.JSONDecodeError:
            pass
    return summary


def _extract_validation_summary(outputs_root: Path, ckpt: Path) -> dict[str, Any]:
    # Prefer explicit checkpoint_validation artifacts that mention this checkpoint.
    for val_path in sorted(outputs_root.glob("checkpoint_validation*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(val_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        checkpoint_str = str(payload.get("checkpoint", ""))
        if checkpoint_str and Path(checkpoint_str).expanduser().resolve() == ckpt.resolve():
            return {
                "artifact": str(val_path.resolve()),
                "is_compatible": payload.get("compatibility", {}).get("is_compatible"),
                "is_legacy": payload.get("compatibility", {}).get("is_legacy"),
                "message": payload.get("compatibility", {}).get("message"),
            }
    return {}


def _extract_evaluation_summary(outputs_root: Path, ckpt: Path) -> dict[str, Any]:
    # If a metrics.json exists under an eval folder modified after checkpoint creation, keep latest.
    metrics_files = sorted(outputs_root.glob("layoutlmv3_eval*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for metrics_path in metrics_files:
        if metrics_path.stat().st_mtime < ckpt.stat().st_mtime:
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        return {
            "artifact": str(metrics_path.resolve()),
            "metrics": payload,
        }
    return {}


def _select_best_validated(entries: list[CheckpointEntry]) -> str:
    if not entries:
        return ""

    def score(entry: CheckpointEntry) -> tuple[int, float, float]:
        schema_score = 2 if entry.schema_status == "richer_schema" else (1 if entry.schema_status == "legacy_reduced_schema" else 0)
        eval_metric = 0.0
        metrics = entry.evaluation_summary.get("metrics") if isinstance(entry.evaluation_summary, dict) else None
        if isinstance(metrics, dict):
            for key in ["f1", "eval_f1", "micro_f1", "macro_f1"]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eval_metric = float(val)
                    break
        return (schema_score, eval_metric, entry.modified_time)

    best = max(entries, key=score)
    return best.path
