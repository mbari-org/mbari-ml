"""Local image access for the ROI review GUI: ROI blob decode + a bounded
LRU cache for full-frame preview images.

This replaces vars-gridview's ``services/roi_service.py``/``roi_loader.py``,
which fetch pixels over HTTP from MBARI's Skimmer/Beholder microservices and
have no image cache at all (see ``THIRD_PARTY_NOTICES.md`` for the parts of
this GUI that *are* adapted from that project). Neither is needed here: an
ROI's pixels are already a JPEG blob sitting in the ``predictions.roi``
column (decode-only, no fetch), and the full source image is a local file on
disk. The old GUI's ``self.full_image_cache = {}`` grew without bound for
the life of the window; this gives it a cap.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np

from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_FULL_IMAGE_CACHE_SIZE = 32


def _blank_roi(size: int = 100) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


class RoiService:
    """Decodes ROI blobs and caches full-frame images, bounded by an LRU."""

    def __init__(self, max_cache_entries: int = DEFAULT_FULL_IMAGE_CACHE_SIZE) -> None:
        self._max_cache_entries = max(1, int(max_cache_entries))
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    def decode_roi(self, roi_blob: bytes | None) -> np.ndarray:
        """Decode a JPEG ROI blob into a BGR ``np.ndarray``.

        Safe to call from a worker thread. Returns a blank placeholder image
        (never ``None``) if the blob is missing or fails to decode, so
        callers don't need a separate null-check path.
        """
        if not roi_blob:
            return _blank_roi()
        try:
            buffer = np.frombuffer(roi_blob, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception:
            logger.exception("Error decoding ROI blob")
            return _blank_roi()
        if image is None:
            logger.warning("Failed to decode ROI blob (cv2.imdecode returned None)")
            return _blank_roi()
        return image

    def fetch_full_image(self, image_path: str) -> np.ndarray | None:
        """Return the full source image for *image_path* as BGR, cached.

        Returns ``None`` if the file can't be read. Safe to call from a
        worker thread.
        """
        with self._lock:
            cached = self._cache.get(image_path)
            if cached is not None:
                self._cache.move_to_end(image_path)
                return cached

        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Could not read image: %s", image_path)
            return None

        with self._lock:
            self._cache[image_path] = image
            self._cache.move_to_end(image_path)
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)

        return image

    def invalidate(self, image_path: str | None = None) -> None:
        """Drop one cached image, or every cached image if *image_path* is None."""
        with self._lock:
            if image_path is None:
                self._cache.clear()
            else:
                self._cache.pop(image_path, None)


def crop_and_encode(
    full_image_bgr: np.ndarray, x_min: float, y_min: float, x_max: float, y_max: float
) -> bytes | None:
    """Crop a region out of a full BGR image and JPEG-encode it.

    The inverse of :meth:`RoiService.decode_roi`: used to regenerate the
    stored ``roi`` blob after a detail-view box edit changes a detection's
    geometry. Coordinates are clamped to the image bounds (a box can be
    dragged right up to the edge) with a minimum-size guard against a
    degenerate (zero-area) crop. Returns ``None`` if the resulting region is
    empty or encoding fails.
    """
    height, width = full_image_bgr.shape[:2]
    x0 = max(0, min(int(round(x_min)), width - 1))
    y0 = max(0, min(int(round(y_min)), height - 1))
    x1 = max(x0 + 1, min(int(round(x_max)), width))
    y1 = max(y0 + 1, min(int(round(y_max)), height))

    crop = full_image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        logger.warning("Degenerate crop (%s,%s)-(%s,%s); not re-encoding", x_min, y_min, x_max, y_max)
        return None

    ok, buffer = cv2.imencode(".jpg", crop)
    if not ok:
        logger.warning("Failed to JPEG-encode cropped ROI")
        return None
    return buffer.tobytes()


__all__ = ["RoiService", "DEFAULT_FULL_IMAGE_CACHE_SIZE", "crop_and_encode"]
