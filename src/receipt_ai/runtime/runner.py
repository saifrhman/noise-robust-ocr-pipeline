from __future__ import annotations

import random
from pathlib import Path

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.pipelines.entrypoints import run_easyocr_rules, run_hybrid, run_layoutlm_only
from src.receipt_ai.runtime.policy import RuntimePolicy, apply_runtime_policy, load_runtime_policy


MODE_LABELS = {
    "easyocr_rules": "EasyOCR + Rules",
    "layoutlm_only": "LayoutLMv3 Only",
    "hybrid": "Hybrid Extraction",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_runtime_config(policy_path: str | Path | None = None) -> tuple[ReceiptAIConfig, RuntimePolicy]:
    cfg = ReceiptAIConfig.from_env()
    policy = load_runtime_policy(policy_path)
    configure_determinism(policy.deterministic_seed)
    return apply_runtime_policy(cfg, policy), policy


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_mode(
    requested_mode: str,
    *,
    checkpoint_override: str = "",
    cfg: ReceiptAIConfig,
    policy: RuntimePolicy,
) -> tuple[str, str, list[str]]:
    if requested_mode == "auto":
        requested_mode = policy.default_mode
    if requested_mode not in MODE_LABELS:
        requested_mode = policy.default_mode if policy.default_mode in MODE_LABELS else "easyocr_rules"

    if requested_mode == "easyocr_rules":
        return requested_mode, "", []

    messages: list[str] = []
    fallback_mode = policy.fallback_mode_on_model_failure or "easyocr_rules"
    raw_checkpoint = (checkpoint_override or "").strip() or str(policy.preferred_checkpoint or "").strip()

    if not raw_checkpoint:
        messages.append(
            f"No fine-tuned LayoutLMv3 checkpoint is configured. Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, "", messages

    if raw_checkpoint == "microsoft/layoutlmv3-base":
        messages.append(
            "microsoft/layoutlmv3-base is not a supported receipt extraction checkpoint. "
            f"Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, "", messages

    checkpoint_path = Path(raw_checkpoint).expanduser()
    if not checkpoint_path.exists():
        messages.append(
            f"Configured checkpoint is missing: {checkpoint_path}. "
            f"Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, "", messages

    compatibility = inspect_checkpoint_label_space(checkpoint_path)
    if (not compatibility.is_compatible) and (not compatibility.is_legacy):
        messages.append(
            f"Checkpoint incompatible: {compatibility.message} "
            f"Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, "", messages

    if compatibility.is_legacy and not bool(policy.allow_legacy_checkpoint):
        messages.append(
            "Legacy/reduced-schema checkpoint detected and disabled by default config. "
            f"Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, "", messages

    if compatibility.is_legacy:
        messages.append("Legacy checkpoint detected; richer-schema extraction may be limited.")

    resolved_checkpoint = str(checkpoint_path.resolve())
    cfg.paths.model_checkpoint = Path(resolved_checkpoint)
    return requested_mode, resolved_checkpoint, messages


def run_extraction(image_path: str | Path, *, mode: str, cfg: ReceiptAIConfig):
    runner = {
        "easyocr_rules": run_easyocr_rules,
        "layoutlm_only": run_layoutlm_only,
        "hybrid": run_hybrid,
    }[mode]
    return runner(image_path, config=cfg)


def iter_input_images(input_path: str | Path) -> list[Path]:
    root = Path(input_path).expanduser().resolve()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Input path not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
