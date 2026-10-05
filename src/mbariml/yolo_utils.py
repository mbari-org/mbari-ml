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

from pathlib import Path

import typer

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)


def read_names_file(names_file: Path) -> list[str]:
    """Class names, in class-index order, from a dataset's ``names.txt`` (one
    per line) or Ultralytics dataset YAML (``names:`` as a list, or as the
    ``{index: name}`` mapping Ultralytics also accepts).

    Shared by ``export yolo --names-file`` (keep an existing dataset's class
    order) and ``import yolo`` (turn class indices back into names), so the
    two sides of the round trip read the same file the same way.
    """
    if names_file.suffix.lower() in (".yaml", ".yml"):
        import yaml  # PyYAML, already installed as an Ultralytics dependency

        names = (yaml.safe_load(names_file.read_text()) or {}).get("names")
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names)]
        if not isinstance(names, list):
            raise typer.BadParameter(f"{names_file}: no `names:` list or mapping found")
        names = [str(name).strip() for name in names]
    else:
        names = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise typer.BadParameter(f"{names_file} lists these names more than once: {duplicates}")
    return names


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
