"""Shared YOLO model loading and device selection.

Every step used to hardcode ``device="mps"``. On a machine without Apple
Silicon (or without a working MPS backend), that either raises deep inside
Ultralytics/torch with a confusing error, or -- worse -- gets silently caught
by a broad ``except Exception`` a few frames up and logged as one line buried
in console output, which is another way to get a "ran but nothing happened"
result. ``resolve_device`` checks hardware availability up front and fails
with a clear, actionable message (or auto-picks the best available device).
"""

from __future__ import annotations

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``requested`` ('auto', 'mps', 'cuda', or 'cpu') against the
    hardware actually available on this machine."""
    import torch

    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Requested device 'mps' but the MPS backend is not available on this machine. "
            "Use --device auto or --device cpu instead."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested device 'cuda' but no CUDA GPU is available on this machine. "
            "Use --device auto or --device cpu instead."
        )
    return requested


def load_model(model_path: str):
    """Load a YOLO model, raising a clear error instead of a raw traceback
    from deep inside Ultralytics if the path is bad.

    Deliberately does NOT pre-validate the path: a stock Ultralytics name
    like 'yolov8n.pt' has a .pt suffix and won't exist locally until
    Ultralytics downloads it on first use, so a suffix/existence check can't
    tell a valid stock alias apart from a typo'd local path. YOLO() itself
    already knows how to resolve either case; only its own failure is worth
    wrapping with a clearer message.
    """
    from ultralytics import YOLO

    logger.info("Loading YOLO model: %s", model_path)
    try:
        return YOLO(model_path)
    except Exception as exc:  # pragma: no cover - message wrapping only
        raise RuntimeError(f"Failed to load YOLO model '{model_path}': {exc}") from exc
