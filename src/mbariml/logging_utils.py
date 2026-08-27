"""Shared logging setup.

The original scripts used a mix of bare ``print`` and ``typer.echo`` calls,
and swallowed exceptions with a plain ``print(f"Error ...: {e}")`` that
discarded the traceback. That made real failures easy to miss in a wall of
YOLO/console output, and impossible to debug after the fact. Every step now
goes through this module instead, so failures are always visible, timestamped,
and (for unexpected exceptions) include a full traceback.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.environ.get("LASSML_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
