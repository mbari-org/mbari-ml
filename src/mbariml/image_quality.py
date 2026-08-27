"""Shared image-quality metrics."""

from __future__ import annotations

import cv2
import numpy as np


def compute_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian -- a standard, cheap blur/sharpness proxy.

    Higher is sharper; blurry or low-detail crops score close to zero. This
    replaces the ``0.0`` placeholder that used to be written for every ROI.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
