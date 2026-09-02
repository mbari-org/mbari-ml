"""Full-image pan/zoom viewer with editable bounding-box overlays.

Adapted from MBARI vars-gridview's pyqtgraph-based detail pane
(``BoxHandler``/``DetailPaneCoordinator`` in ``ui/coordinators/``; MIT
License, see ``THIRD_PARTY_NOTICES.md``). Wheel-zoom and drag-pan are
entirely default ``pyqtgraph.ViewBox`` behavior -- no custom mouse-event
code needed here, matching the original, which has none either -- except for
"draw mode" (see ``_DrawableViewBox``), added for the "Add New ROI" tool,
which is new here (not present in vars-gridview).

Dropped relative to vars-gridview: the dirty-flag/deferred-save machinery
and the dedicated background thread pool for image loading. This class
assumes its caller already has a decoded image in hand (see
``RoiService.fetch_full_image``, already cached) and that box edits persist
immediately (see ``bounding_box.py``), so there's nothing here to defer or
batch.
"""

from __future__ import annotations

import gc
from collections.abc import Callable

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from mbariml.gui.bounding_box import BoundingBox
from mbariml.gui.query_service import FrameRoi

# Row-major (numpy's natural H x W x C layout), not pyqtgraph's column-major
# default -- must be set before any ImageItem is shown, or images render
# transposed.
pg.setConfigOptions(imageAxisOrder="row-major")

# Minimum drag size (in image pixels) to count as a real box, not an
# accidental click/jitter -- same idea as BoundingBox's own 1px minimum size
# guard in bounding_box.py.
MIN_DRAWN_BOX_SIZE = 3


class _DrawableViewBox(pg.ViewBox):
    """A ``ViewBox`` that behaves exactly like the plain default (wheel-zoom,
    drag-to-pan) unless ``draw_mode`` is on, in which case a left-button drag
    draws a rubber-band rectangle instead of panning, and reports the
    finished rectangle -- in image-pixel coordinates -- via ``box_drawn``.

    Reuses ``ViewBox``'s own built-in ``RectMode`` scale-box mechanics
    (``updateScaleBox``/``rbScaleBox``, and the same
    ``childGroup.mapRectFromParent`` coordinate mapping ``RectMode`` itself
    uses on release -- see ``pyqtgraph.graphicsItems.ViewBox.ViewBox.
    mouseDragEvent``) purely for the live selection-box visuals and the
    scene-to-image-pixel coordinate math, not for the "zoom to the drawn
    rect" behavior ``RectMode`` normally triggers on release -- that's
    replaced with the ``box_drawn`` callback instead, and the view's zoom
    level is left untouched.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.draw_mode = False
        self.box_drawn: Callable[[float, float, float, float], None] | None = None

    def mouseDragEvent(self, ev, axis=None) -> None:  # noqa: N802 (pyqtgraph's own naming)
        if not self.draw_mode or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis=axis)
            return

        ev.accept()
        if ev.isFinish():
            self.rbScaleBox.hide()
            rect = QtCore.QRectF(ev.buttonDownPos(ev.button()), ev.pos())
            rect = self.childGroup.mapRectFromParent(rect).normalized()
            if (
                self.box_drawn is not None
                and rect.width() >= MIN_DRAWN_BOX_SIZE
                and rect.height() >= MIN_DRAWN_BOX_SIZE
            ):
                self.box_drawn(rect.left(), rect.top(), rect.right(), rect.bottom())
        else:
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())


class DetailView(QtWidgets.QWidget):
    """Full-image panel: wheel-zoom/drag-pan plus editable detection boxes."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._graphics_view = pg.GraphicsView(self)
        # pyqtgraph's own widgets aren't reached by Qt stylesheets at all --
        # its default is a light gray background, which would otherwise be
        # the one bright panel left over in an all-dark window.
        self._graphics_view.setBackground("#1e1f22")
        self._view_box = _DrawableViewBox()
        self._view_box.setAspectLocked()
        self._view_box.invertY(True)  # match image pixel coords (y grows downward)
        self._graphics_view.setCentralItem(self._view_box)

        self._image_item = pg.ImageItem(autoDownsample=True)
        self._view_box.addItem(self._image_item)

        self._placeholder = pg.TextItem("Select an ROI to view the full image.", anchor=(0.5, 0.5))
        self._view_box.addItem(self._placeholder)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._graphics_view)

        self._current_image_path: str | None = None
        self._image_size: tuple[int, int] | None = None  # (width, height)
        self._boxes: list[BoundingBox] = []

    def show_image(self, image_path: str, image_bgr: np.ndarray) -> None:
        """Display *image_bgr*, auto-ranging only for a genuinely new image
        so re-coloring boxes on an already-shown frame doesn't reset the
        user's current zoom/pan."""
        if self._placeholder is not None:
            self._view_box.removeItem(self._placeholder)
            self._placeholder = None

        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._image_item.setImage(rgb)

        is_new_image = image_path != self._current_image_path
        self._current_image_path = image_path
        self._image_size = (width, height)
        if is_new_image:
            self._view_box.autoRange()

    def clear(self) -> None:
        self._clear_boxes()
        self._image_item.clear()
        self._current_image_path = None
        self._image_size = None

    def set_draw_mode(
        self, enabled: bool, on_box_drawn: Callable[[float, float, float, float], None] | None = None
    ) -> None:
        """Turn "Add New ROI" drawing on/off for the ``ViewBox`` (see
        ``_DrawableViewBox``). While on, a left-drag on empty image
        background draws a new box instead of panning, reported via
        *on_box_drawn* as ``(x_min, y_min, x_max, y_max)`` in image-pixel
        coordinates; wheel-zoom and dragging an *existing* box (which
        intercepts the drag itself, before it ever reaches the ViewBox) both
        keep working exactly as before. A crosshair cursor is the only other
        visible sign draw mode is active -- there's no separate "drawing
        overlay" to tear down on exit.
        """
        self._view_box.draw_mode = enabled
        self._view_box.box_drawn = on_box_drawn if enabled else None
        self._graphics_view.setCursor(
            QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor
        )

    def set_boxes(
        self,
        frame_rois: list[FrameRoi],
        active_roi_indices: set[int],
        *,
        on_clicked: Callable[[int, object], None],
        on_changed: Callable[[int, float, float, float, float], None],
        on_delete: Callable[[int], None],
    ) -> None:
        """Rebuild the box overlays for the currently-shown image -- one per
        detection on that frame, regardless of which mosaic page it's on."""
        self._clear_boxes()
        if self._image_size is None:
            return
        width, height = self._image_size
        max_bounds = QtCore.QRectF(0, 0, width, height)

        for frame_roi in frame_rois:
            box = BoundingBox(
                frame_roi,
                self._view_box,
                max_bounds=max_bounds,
                is_active=frame_roi.roi_index in active_roi_indices,
                clicked_callback=on_clicked,
                changed_callback=on_changed,
                delete_callback=on_delete,
            )
            self._boxes.append(box)

    def update_active(self, active_roi_indices: set[int]) -> None:
        """Recolor existing boxes in place (no rebuild) -- e.g. on grid
        selection change, when the set of detections hasn't changed."""
        for box in self._boxes:
            box.set_active(box.frame_roi.roi_index in active_roi_indices)

    def update_label(self, roi_index: int, label: str) -> None:
        """Reflect a label change (applied via the controls panel) onto an
        already-drawn box's caption, without rebuilding it."""
        for box in self._boxes:
            if box.frame_roi.roi_index == roi_index:
                box.frame_roi.label = label
                box.draw_name()

    def remove_box(self, roi_index: int) -> None:
        """Drop one overlay (its ROI was deleted)."""
        remaining = []
        for box in self._boxes:
            if box.frame_roi.roi_index == roi_index:
                box.remove()
            else:
                remaining.append(box)
        self._boxes = remaining

    def _clear_boxes(self) -> None:
        for box in self._boxes:
            box.remove()
        self._boxes = []

        # pyqtgraph's ROI/RectROI keeps internal references (e.g. a
        # MouseDragHandler per handle) that cycle back to the ROI itself --
        # confirmed directly: even after remove() detaches from the scene
        # and clears our own callbacks, a box needs the *cyclic* GC to be
        # collected, not plain refcounting. That's a hazard for a
        # QGraphicsItem specifically (see RectWidget.cleanup()'s docstring
        # for why, traced to an actual SIGSEGV): the cyclic GC runs at an
        # unpredictable moment, and finalizing a QGraphicsItem while another
        # one is mid-construction elsewhere is what crashed. Forcing
        # collection right here -- right after detaching, right before
        # set_boxes() goes on to construct new BoundingBox instances -- means
        # it happens at a moment we control instead.
        gc.collect()


__all__ = ["DetailView"]
