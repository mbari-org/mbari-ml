"""Setup: fetch and check the optional SAM3 model the review GUI can use.

Not a pipeline stage -- SAM3 only assists ``mbariml review`` (one-click boxes,
box tightening; see ``mbariml.gui.sam3_service``). These two commands replace
hand-placing ``sam3.pt`` and pointing review at it:

    mbariml sam3 download   sam3.pt into the Hugging Face cache, where
                            review finds it with no --sam3-model or env var
    mbariml sam3 check      which model review would use, or what's missing
"""

from __future__ import annotations

import os

import typer

from mbariml.gui import sam3_service


def download() -> None:
    """Download SAM3 (sam3.pt, ~3.4 GB) from Hugging Face for the review GUI.

    The repo is gated: request access at https://huggingface.co/facebook/sam3,
    then log in once with `hf auth login` (or set HF_TOKEN). The file goes to
    the Hugging Face cache, where `mbariml review` finds it by itself. Run it
    again any time: an existing download is reused, not fetched again.
    """
    try:
        path = sam3_service.download_model()
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    typer.echo(f"SAM3 model: {path}")
    _report(path)


def check(
    sam3_model: str = typer.Option(
        None, "--sam3-model", help="Check this sam3.pt instead of the one review would find by itself."
    ),
) -> None:
    """Show which SAM3 model `mbariml review` would use, or what's missing.

    Looks where review looks: --sam3-model, then $MBARIML_SAM3_MODEL, then the
    Hugging Face cache (from `mbariml sam3 download`). Only checks files and
    packages; does not load the model. Exits 1 if SAM3 isn't usable.
    """
    path = sam3_service.resolve_model_path(sam3_model)
    if sam3_model:
        source = "--sam3-model"
    elif path and path == os.environ.get(sam3_service.MODEL_ENV_VAR):
        source = f"${sam3_service.MODEL_ENV_VAR}"
    elif path:
        source = "Hugging Face cache"
    else:
        source = None
    typer.echo(f"SAM3 model: {path} ({source})" if path else "SAM3 model: none found")
    if not _report(path):
        raise typer.Exit(1)


def _report(path: str | None) -> bool:
    reason = sam3_service.unavailable(path)
    typer.echo(f"Not usable: {reason}" if reason else "Ready: `mbariml review` will enable its SAM3 controls.")
    return reason is None
