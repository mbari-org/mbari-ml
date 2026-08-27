"""Deterministic label -> color mapping for tile borders/labels.

Adapted from MBARI vars-gridview's ``lib/vision/image_utils.py`` (MIT
License; see ``THIRD_PARTY_NOTICES.md``) -- just the ``color_for_concept``
hash helper. That module's ``fetch_image`` (HTTP/Beholder frame fetch) has
no equivalent here since ROI pixels never leave the local DuckDB file.
"""

from __future__ import annotations

from functools import cache

from PySide6.QtGui import QColor


@cache
def color_for_label(label: str) -> QColor:
    """Return a stable HSL colour derived from the label string."""
    hue_raw = sum(ord(c) for c in label) << 5
    color = QColor()
    color.setHsl(round((hue_raw % 360) / 360 * 255), 255, 217, 255)
    return color


__all__ = ["color_for_label"]
