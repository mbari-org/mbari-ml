"""Draggable/resizable bounding-box overlay for the full-image detail view.

Adapted from MBARI vars-gridview's ``ui/mosaic/bounding_box.py`` (MIT
License; see ``THIRD_PARTY_NOTICES.md``). The pyqtgraph ``RectROI``
mechanics -- handle setup, ``maxBounds`` clamping to the image rect,
region-changed signals -- are generic and port essentially unchanged; only
what happens *on* an edit is different: vars-gridview marks a VARS
``BoundingBoxAssociation`` dirty and defers a batched HTTP push through its
``AnnotationService`` (necessary there because Annosaurus writes are slow
network calls worth coalescing). Here, edits persist immediately on
drag-release -- a local DuckDB UPDATE is fast enough that deferring it would
only add risk (an edit lost to a crash before the deferred save runs) for no
benefit. The "Change concept"/"Change part" context-menu actions are also
dropped: there's no concept/part hierarchy in mbariml, just a free-text
``new_label`` column, and labeling stays in the controls panel.
"""

from __future__ import annotations

from collections.abc import Callable
from html import escape

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from mbariml.gui.query_service import FrameRoi

# Per the user's request: the active (selected) box is red, every other
# detection on the same frame is a pale, unobtrusive blue-grey.
ACTIVE_COLOR = QtGui.QColor("#ff3b30")
INACTIVE_COLOR = QtGui.QColor(170, 190, 210, 200)
PEN_WIDTH = 2


class BoundingBox(pg.RectROI):
    """One draggable/resizable overlay box, backed by a single ``FrameRoi``."""

    def __init__(
        self,
        frame_roi: FrameRoi,
        view: pg.ViewBox,
        *,
        max_bounds: QtCore.QRectF,
        is_active: bool,
        clicked_callback: Callable[[int, object], None],
        changed_callback: Callable[[int, float, float, float, float], None],
        delete_callback: Callable[[int], None],
    ) -> None:
        self.frame_roi = frame_roi
        self._clicked_callback = clicked_callback
        self._changed_callback = changed_callback
        self._delete_callback = delete_callback

        pos = (frame_roi.x_min, frame_roi.y_min)
        size = (
            max(1.0, frame_roi.x_max - frame_roi.x_min),
            max(1.0, frame_roi.y_max - frame_roi.y_min),
        )
        pen = pg.mkPen(ACTIVE_COLOR if is_active else INACTIVE_COLOR, width=PEN_WIDTH)

        pg.RectROI.__init__(
            self,
            pos,
            size,
            pen=pen,
            invertible=True,
            rotatable=False,
            removable=False,
            sideScalers=True,
            maxBounds=max_bounds,
        )

        # sideScalers=True only auto-adds two of the eight handles; add the rest.
        self.addScaleHandle([1, 0], [0, 1])
        self.addScaleHandle([0, 1], [1, 0])
        self.addScaleHandle([0, 0], [1, 1])
        self.addScaleHandle([0, 0.5], [1, 0.5])
        self.addScaleHandle([0.5, 0], [0.5, 1])
        self.addTranslateHandle([0.5, 0.5])

        self.view = view
        self.view.addItem(self)

        self.text_item = pg.TextItem(anchor=(0, 1))
        self.view.addItem(self.text_item)
        self.draw_name()

        self.sigRegionChanged.connect(self.draw_name)
        self.sigRegionChangeFinished.connect(self._on_region_change_finished)
        self.sigClicked.connect(self._on_clicked)

    def set_active(self, is_active: bool) -> None:
        """Recolor without rebuilding -- called when grid selection changes."""
        self.setPen(pg.mkPen(ACTIVE_COLOR if is_active else INACTIVE_COLOR, width=PEN_WIDTH))

    def draw_name(self) -> None:
        """Two stacked lines: curated new_label (bold) and raw
        original_label (plain), both shown always -- no display-mode toggle
        to pick just one (mirrors RectWidget.paint()).

        ``anchor=(0, 1)`` pins the *bottom-left* of the whole rendered block
        to the box's position, so the block grows upward from there at a
        fixed pixel height regardless of zoom. For a box near the top of
        the image, that upward growth can push part of the block above the
        viewport -- and with two lines that block is twice as tall as the
        single-line original, so it clips far more often. Confirmed
        directly: rendering a box near the top edge clipped the topmost
        line entirely while the line right above the anchor stayed visible.
        The HTML div written *last* ends up closest to the anchor (least
        likely to clip), so new_label -- the one that actually matters --
        goes last/bottom here, with original_label taking the clipping risk
        instead.
        """
        x, y, _w, _h = self.get_box()
        new_label = self.frame_roi.label
        original_label = self.frame_roi.original_label
        top = escape(original_label) if original_label else "&nbsp;"
        bottom = escape(new_label) if new_label else "&nbsp;"
        self.text_item.setHtml(
            f'<div style="color:#cccccc; font-weight:normal; font-size:9pt;">{top}</div>'
            f'<div style="color:white; font-weight:bold; font-size:11pt;">{bottom}</div>'
        )
        self.text_item.setPos(x, y)

    def get_box(self) -> tuple[float, float, float, float]:
        """Current geometry as ``(x, y, width, height)``, normalized to
        non-negative width/height (``invertible=True`` lets a drag briefly
        produce a negative size when a handle crosses the opposite edge)."""
        pos = self.pos()
        size = self.size()
        x, y, w, h = pos.x(), pos.y(), size.x(), size.y()
        if w < 0:
            x += w
            w = -w
        if h < 0:
            y += h
            h = -h
        return x, y, w, h

    def remove(self) -> None:
        """Detach both graphics items (box + label) from the view, and break
        the reference cycle back to the owning window.

        ``_clicked_callback``/``_changed_callback``/``_delete_callback`` are
        bound methods of ``MainWindow`` -- so every box forms a cycle
        (window -> DetailView -> box -> bound method -> window) that plain
        refcounting can never break; it needs Python's cyclic GC, which runs
        at an unpredictable moment. For a QGraphicsItem that's a real
        hazard, not just a delay -- see the identical fix (and its
        docstring) on RectWidget.cleanup(), which traces this exact pattern
        to an actual SIGSEGV. Clearing the callbacks here makes a removed
        box plain acyclic garbage, collected immediately instead.
        """
        try:
            self.sigClicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            self.sigRegionChanged.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            self.sigRegionChangeFinished.disconnect()
        except (RuntimeError, TypeError):
            pass
        self._clicked_callback = None
        self._changed_callback = None
        self._delete_callback = None
        self.view.removeItem(self.text_item)
        self.view.removeItem(self)

    def _on_region_change_finished(self) -> None:
        x, y, w, h = self.get_box()
        self._changed_callback(self.frame_roi.roi_index, x, y, x + w, y + h)

    def _on_clicked(self, _roi, event) -> None:
        self._clicked_callback(self.frame_roi.roi_index, event)

    def contextMenuEvent(self, event) -> None:
        menu = QtWidgets.QMenu()
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_callback(self.frame_roi.roi_index))
        menu.exec(event.screenPos())


__all__ = ["BoundingBox", "ACTIVE_COLOR", "INACTIVE_COLOR"]
