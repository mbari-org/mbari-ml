"""Single-ROI tile widget for the image mosaic grid.

Adapted from MBARI vars-gridview's ``ui/mosaic/rect_widget.py`` (MIT
License; see ``THIRD_PARTY_NOTICES.md``), translated from PyQt6 to PySide6
and trimmed to mbariml's flat ``predictions`` schema: one tile is one
``RoiRow`` (see ``mbariml.gui.query_service``), not an "association" within
a VARS observation/imaged-moment hierarchy. Dropped relative to the
original: video/image-reference source resolution, ancillary sensor data,
training flags/menu actions, and the live per-tile HTTP embedding model --
mbariml precomputes embeddings in ``mbariml embed`` and stores them on the
row, so similarity search is a DuckDB query
(``query_service.compute_similarity_order``) over stored vectors, not
something a loaded tile needs to compute for itself. A ``verified`` flag
*is* kept, though (see ``set_verified()`` / the badge in ``paint()``) --
that one's mbariml's own, not a straight port.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QThreadPool, Signal, Slot

from mbariml.gui.colors import color_for_label
from mbariml.gui.query_service import RoiRow
from mbariml.gui.roi_service import RoiService
from mbariml.gui.runnables import Worker
from mbariml.logging_utils import get_logger

logger = get_logger(__name__)

THUMB_SIZE = 120
SELECTION_HIGHLIGHT_COLOR = QtGui.QColor("#34a1eb")
VERIFIED_BADGE_COLOR = QtGui.QColor("#2ecc71")


def _readable_text_color(background: QtGui.QColor) -> QtGui.QColor:
    """Black or white, whichever reads better against *background*.

    Standard relative-luminance threshold, not a fixed color: a tile's fill
    is either a light pastel (``color_for_label()``, when labeled) or a
    dark navy fallback (unlabeled) -- one fixed text color can't read on
    both.
    """
    luminance = 0.299 * background.red() + 0.587 * background.green() + 0.114 * background.blue()
    return QtGui.QColor(QtCore.Qt.GlobalColor.black if luminance > 140 else QtCore.Qt.GlobalColor.white)


class RectWidget(QtWidgets.QGraphicsWidget):
    roiRefreshed = Signal(object)  # self
    clicked = Signal(object, object)  # self, event
    similaritySort = Signal(object, bool)  # self, same_label_only

    def __init__(
        self,
        row: RoiRow,
        roi_service: RoiService,
        clicked_slot: Callable,
        similarity_sort_slot: Callable,
        parent=None,
        zoom: float = 1.0,
        preload_roi: bool = False,
        brightness: int = 0,
        contrast: float = 1.0,
    ) -> None:
        QtWidgets.QGraphicsWidget.__init__(self, parent)

        self.row = row
        self._roi_service = roi_service
        self._zoom = max(float(zoom), 0.01)
        # Display-only brightness/contrast, applied when the thumbnail pixmap
        # is built (see getpic). Never touches self.roi or anything stored --
        # same "view constraint only" rule as the min-confidence filter.
        self._brightness = int(brightness)
        self._contrast = float(contrast)

        # Tall enough for two stacked label lines (new_label on top, bold;
        # original_label below, plain -- see paint()) -- both are always
        # shown now, there's no display-mode toggle to pick just one.
        self.labelheight = 34
        self.bordersize = 4
        self.outlinesize = 6
        self.picdims = [THUMB_SIZE, THUMB_SIZE]
        self.background_color = QtGui.QColor.fromRgb(25, 35, 45)

        self.is_selected = False

        self.roi: np.ndarray | None = None
        self.pic: QtGui.QPixmap | None = None
        if preload_roi:
            self.update_roi_pic()
        else:
            # Paint a cheap placeholder immediately; the mosaic's loading
            # coordinator fetches the real ROI asynchronously right after.
            self.roi = np.zeros((self.picdims[1], self.picdims[0], 3), dtype=np.uint8)
            self.pic = self.getpic(self.roi)

        self._clicked_slot = clicked_slot
        self._similarity_sort_slot = similarity_sort_slot
        self.clicked.connect(self._clicked_slot)
        self.similaritySort.connect(self._similarity_sort_slot)

        self._roi_refresh_generation = 0
        self._roi_batch_generation = 0

        # QRunnable (unlike QObject) gets no parent-child ownership tracking
        # to keep its Python wrapper alive once QThreadPool takes it -- with
        # nothing else referencing a dispatched Worker, CPython can garbage
        # collect it (and its `signals` QObject) while the pool is still
        # running it on another thread, which crashes. Keeping every
        # dispatched Worker here for this tile's lifetime is the standard
        # fix; the memory cost is negligible (a handful of tiny objects per
        # tile, not thousands).
        self._roi_workers: list[Worker] = []

    # -- Identity -----------------------------------------------------------

    @property
    def roi_index(self) -> int:
        return self.row.roi_index

    @property
    def label(self) -> str | None:
        return self.row.label

    def set_label(self, label: str) -> None:
        """Update this tile's label in place (no ROI re-decode, no rebuild)."""
        self.row.label = label
        self.update()

    def set_verified(self, verified: bool) -> None:
        """Update this tile's verified flag in place (no ROI re-decode, no
        rebuild) -- just flips the badge drawn by ``paint()``."""
        self.row.verified = verified
        self.update()

    @property
    def roi_batch_generation(self) -> int:
        """Current ROI loading batch generation assigned by the loading coordinator."""
        return self._roi_batch_generation

    def assign_roi_batch_generation(self, generation: int) -> None:
        self._roi_batch_generation = generation

    def cleanup(self) -> None:
        """Break the reference cycle back to the owning window.

        ``self._clicked_slot``/``self._similarity_sort_slot`` are bound
        methods of ``MainWindow``, and the ``clicked``/``similaritySort``
        signal connections themselves also hold a reference to those bound
        methods -- so every tile forms a cycle (window -> tile -> bound
        method -> window) that plain refcounting can never break. Cycles
        like that only get collected by Python's *cyclic* GC, which runs at
        an unpredictable moment; for a QGraphicsItem that's a real hazard,
        not just a delay -- confirmed by an actual SIGSEGV whose native
        stack trace showed the GC firing reentrantly during construction of
        an unrelated new QGraphicsItem, finalizing a batch of these cycles,
        and crashing inside a QGraphicsItem destructor's Qt/Python
        round-trip. Call this on every tile at the moment it's discarded
        (see MainWindow's discard points) so it becomes plain acyclic
        garbage, collected immediately and deterministically instead.
        """
        try:
            self.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        try:
            self.similaritySort.disconnect()
        except (RuntimeError, TypeError):
            pass
        self._clicked_slot = None
        self._similarity_sort_slot = None

    # -- ROI loading ----------------------------------------------------------

    def get_roi(self) -> np.ndarray:
        return self._roi_service.decode_roi(self.row.roi_blob)

    def update_roi_pic(self) -> None:
        self.roi = self.get_roi()
        self.pic = self.getpic(self.roi)
        self.update()

    def request_roi_refresh(self) -> None:
        """Decode this tile's ROI asynchronously and apply only the latest result."""
        self._roi_refresh_generation += 1
        generation = self._roi_refresh_generation

        worker = Worker(self._decode_with_generation, generation)
        worker.signals.result.connect(self._on_async_roi_refresh_result)
        worker.signals.error.connect(self._on_async_roi_refresh_error)
        self._roi_workers.append(worker)  # see __init__ comment: must outlive the thread pool run
        QThreadPool.globalInstance().start(worker)

    def _decode_with_generation(self, generation: int):
        return generation, self.get_roi()

    def _forget_worker(self, signals_obj: object) -> None:
        """Drop a completed worker from ``_roi_workers``.

        ``_fn`` on each ``Worker`` is a bound method of this very widget
        (``self._decode_with_generation``), so ``_roi_workers`` is a
        self-cycle (widget -> _roi_workers -> Worker -> bound method ->
        widget) for as long as an entry sits in it -- harmless on its own,
        but combined with this being a QGraphicsItem, it's one more thing
        that would otherwise only be collectible by the cyclic GC (see
        ``cleanup()``'s docstring for why that's a real hazard here, not
        just a delay). Safe to prune here: by the time this fires, the
        worker's own ``run()`` has already emitted its result/error, so its
        own executing frame -- not our list -- is what was protecting it
        during the actual decode.
        """
        self._roi_workers = [w for w in self._roi_workers if w.signals is not signals_obj]

    @Slot(object)
    def _on_async_roi_refresh_result(self, payload) -> None:
        self._forget_worker(self.sender())
        generation, roi = payload
        if generation != self._roi_refresh_generation:
            return
        self.roi = roi
        self.pic = self.getpic(roi)
        self.update()
        self.roiRefreshed.emit(self)

    @Slot(tuple)
    def _on_async_roi_refresh_error(self, err: tuple) -> None:
        self._forget_worker(self.sender())
        message = str(err[1]) if len(err) > 1 else "Unknown error"
        logger.error("Error decoding ROI #%s: %s", self.roi_index, message)

        placeholder = self._make_placeholder_roi()
        self.roi = placeholder
        self.pic = self.getpic(placeholder)
        self.update()
        self.roiRefreshed.emit(self)

    # -- Geometry -------------------------------------------------------------

    def _adjusted(self, roi: np.ndarray) -> np.ndarray:
        """Apply the current display brightness/contrast to a thumbnail.

        Contrast pivots around mid-grey (128) rather than around black:
        ``cv2.convertScaleAbs``'s plain ``alpha * pixel + beta`` scales about
        zero, so raising contrast would also wash the whole tile brighter and
        the two sliders would fight each other. Folding ``128 * (1 - alpha)``
        into beta keeps mid-grey fixed, so contrast stretches the range about
        the middle and brightness alone shifts it -- which is what makes them
        usable as two independent controls.
        """
        if self._brightness == 0 and self._contrast == 1.0:
            return roi  # neutral: don't pay for a no-op conversion
        beta = 128.0 * (1.0 - self._contrast) + self._brightness
        return cv2.convertScaleAbs(roi, alpha=self._contrast, beta=beta)

    def set_display_adjustment(self, brightness: int, contrast: float) -> None:
        """Re-render this tile's thumbnail at a new brightness/contrast.

        Rebuilds the pixmap from ``self.roi``, the already-decoded crop, so
        this costs no database read and no JPEG decode -- and leaves the
        stored ROI untouched.
        """
        brightness, contrast = int(brightness), float(contrast)
        if (brightness, contrast) == (self._brightness, self._contrast):
            return
        self._brightness, self._contrast = brightness, contrast
        if self.roi is not None:
            self.pic = self.getpic(self.roi)
            self.update()

    def update_zoom(self, zoom: float) -> None:
        self._zoom = max(float(zoom), 0.01)
        self.boundingRect()
        self.updateGeometry()

    @property
    def outline_x(self):
        return 0

    @property
    def outline_y(self):
        return 0

    @property
    def outline_width(self):
        return self.picdims[0] + self.bordersize * 2 + self.outlinesize * 2

    @property
    def outline_height(self):
        return self.picdims[1] + self.labelheight + self.bordersize * 2 + self.outlinesize * 2

    @property
    def border_x(self):
        return self.outline_x + self.outlinesize

    @property
    def border_y(self):
        return self.outline_y + self.outlinesize

    @property
    def border_width(self):
        return self.outline_width - self.outlinesize * 2

    @property
    def border_height(self):
        return self.outline_height - self.outlinesize * 2

    @property
    def pic_x(self):
        return self.border_x + self.bordersize

    @property
    def pic_y(self):
        return self.border_y + self.bordersize

    @property
    def pic_width(self):
        return self.picdims[0]

    @property
    def pic_height(self):
        return self.picdims[1]

    @property
    def label_x(self):
        return self.pic_x

    @property
    def label_y(self):
        return self.pic_y + self.pic_height

    @property
    def label_width(self):
        return self.pic_width

    @property
    def label_height(self):
        return self.labelheight

    def scale_rect(self, rect: QtCore.QRectF) -> QtCore.QRect:
        return QtCore.QRect(
            round(rect.x() * self._zoom),
            round(rect.y() * self._zoom),
            round(rect.width() * self._zoom),
            round(rect.height() * self._zoom),
        )

    @property
    def outline_rect(self):
        return self.scale_rect(
            QtCore.QRectF(self.outline_x, self.outline_y, self.outline_width, self.outline_height)
        )

    @property
    def border_rect(self):
        return self.scale_rect(
            QtCore.QRectF(self.border_x, self.border_y, self.border_width, self.border_height)
        )

    @property
    def pic_rect(self):
        return self.scale_rect(QtCore.QRectF(self.pic_x, self.pic_y, self.pic_width, self.pic_height))

    @property
    def label_rect(self):
        return self.scale_rect(
            QtCore.QRectF(self.label_x, self.label_y, self.label_width, self.label_height)
        )

    def boundingRect(self):
        return QtCore.QRectF(
            self._zoom * self.outline_x,
            self._zoom * self.outline_y,
            self._zoom * self.outline_width,
            self._zoom * self.outline_height,
        )

    def sizeHint(self, which, constraint=None):
        return self.boundingRect().size()

    # -- Pixmap conversion ------------------------------------------------

    def toqimage(self, img: np.ndarray) -> QtGui.QImage:
        height, width, bytes_per_component = img.shape
        bytes_per_line = bytes_per_component * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return QtGui.QImage(
            img.copy(), width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        )

    def getpic(self, roi: np.ndarray) -> QtGui.QPixmap:
        """Fit *roi* into a ``picdims``-sized square, padded to fill it."""
        max_width = self.pic_width
        max_height = self.pic_height

        if roi is None or roi.ndim < 2 or roi.shape[0] <= 0 or roi.shape[1] <= 0:
            roi = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif roi.ndim == 3 and roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)

        roi_height, roi_width = roi.shape[:2]
        scale = min(max_width / roi_width, max_height / roi_height)
        roi = cv2.resize(roi, (0, 0), fx=scale, fy=scale)

        # Deliberately after the resize: this runs on a ~120x120 thumbnail
        # rather than the full-resolution crop, which is what keeps dragging
        # the sliders across a 500-tile page affordable.
        roi = self._adjusted(roi)

        pad_x = (max_width - roi.shape[1]) // 2
        pad_y = (max_height - roi.shape[0]) // 2
        roi_padded = cv2.copyMakeBorder(
            roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(45, 35, 25)
        )

        return QtGui.QPixmap.fromImage(self.toqimage(roi_padded))

    def _make_placeholder_roi(self) -> np.ndarray:
        width = max(1, int(self.picdims[0]))
        height = max(1, int(self.picdims[1]))
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            placeholder, "unavailable", (6, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )
        return placeholder

    # -- Painting / interaction ---------------------------------------------

    def paint(self, painter, option, widget):
        if self.is_selected:
            painter.fillRect(self.outline_rect, SELECTION_HIGHLIGHT_COLOR)

        # Both labels are always shown now (no display-mode toggle): the
        # curated new_label, when present, drives the tile's fill color --
        # it's the more specific, human-reviewed grouping -- falling back to
        # the raw original_label pre-clustering, when new_label is blank for
        # every row.
        new_label = self.row.label
        original_label = self.row.original_label
        color_key = new_label or original_label
        fill_color = color_for_label(color_key) if color_key else self.background_color
        painter.fillRect(self.border_rect, fill_color)
        painter.fillRect(self.label_rect, fill_color)

        painter.setBackgroundMode(QtCore.Qt.BGMode.TransparentMode)
        if self.pic is not None:
            painter.drawPixmap(self.pic_rect, self.pic, self.pic.rect())

        # Text color must track the fill it's drawn over: color_for_label()'s
        # pastel fills read fine with black text, but the dark "no label"
        # fallback does not -- an earlier revision hardcoded black
        # unconditionally, so every tile was unreadable (black text on a
        # near-black background) whenever new_label was still NULL for
        # everything, i.e. always, before `mbariml cluster` has ever run.
        text_color = _readable_text_color(fill_color)
        painter.setPen(QtGui.QPen(text_color))

        label_rect = self.label_rect
        top_rect = QtCore.QRect(
            label_rect.x(), label_rect.y(), label_rect.width(), label_rect.height() // 2
        )
        bottom_rect = QtCore.QRect(
            label_rect.x(),
            label_rect.y() + top_rect.height(),
            label_rect.width(),
            label_rect.height() - top_rect.height(),
        )

        # Curated label on top, bold; raw YOLO class below, plain and a
        # touch smaller -- so it reads as secondary/provenance info, not a
        # second real label competing with the first.
        #
        # drawText(rect, ...) clips to the rect with no ellipsis of its own:
        # a label wider than the tile (common -- curated labels tend to be
        # longer/more descriptive than the raw YOLO class) got silently cut
        # off mid-word on both sides with no "..." to show it was truncated,
        # which for a sufficiently long label reads as the label just not
        # being there at all. elidedText() truncates it into something that
        # actually fits, with an ellipsis marking that it did.
        top_font = QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold)
        painter.setFont(top_font)
        top_text = QtGui.QFontMetrics(top_font).elidedText(
            new_label or "", QtCore.Qt.TextElideMode.ElideRight, top_rect.width()
        )
        painter.drawText(top_rect, QtCore.Qt.AlignmentFlag.AlignCenter, top_text)

        bottom_font = QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Normal)
        painter.setFont(bottom_font)
        bottom_text = QtGui.QFontMetrics(bottom_font).elidedText(
            original_label or "", QtCore.Qt.TextElideMode.ElideRight, bottom_rect.width()
        )
        painter.drawText(bottom_rect, QtCore.Qt.AlignmentFlag.AlignCenter, bottom_text)

        if self.row.verified:
            self._paint_verified_badge(painter)

    def _paint_verified_badge(self, painter: QtGui.QPainter) -> None:
        """Small green checkmark badge in the thumbnail's upper-right
        corner, marking this ROI as reviewed/verified. Sized off pic_rect
        (already zoom-scaled -- see scale_rect()) so it stays proportional
        as the tile-size slider changes."""
        pic_rect = self.pic_rect
        diameter = max(10, round(pic_rect.height() * 0.22))
        margin = max(2, round(diameter * 0.15))
        badge_rect = QtCore.QRect(
            pic_rect.right() - diameter - margin,
            pic_rect.top() + margin,
            diameter,
            diameter,
        )

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(VERIFIED_BADGE_COLOR))
        painter.drawEllipse(badge_rect)

        check_pen = QtGui.QPen(QtCore.Qt.GlobalColor.white)
        check_pen.setWidthF(max(1.5, diameter * 0.14))
        check_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        check_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(check_pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        center = badge_rect.center()
        cx, cy = float(center.x()), float(center.y())
        r = diameter / 2.0
        p1 = QtCore.QPointF(cx - r * 0.5, cy)
        p2 = QtCore.QPointF(cx - r * 0.1, cy + r * 0.4)
        p3 = QtCore.QPointF(cx + r * 0.55, cy - r * 0.35)
        painter.drawLine(p1, p2)
        painter.drawLine(p2, p3)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self, event)
        else:
            self._handle_right_click(event)

    def _handle_right_click(self, event) -> None:
        menu = QtWidgets.QMenu()

        similarity_sort = menu.addAction("Find similar")
        similarity_sort.triggered.connect(lambda: self.similaritySort.emit(self, False))

        similarity_sort_same_label = menu.addAction("Find similar with same label")
        similarity_sort_same_label.triggered.connect(lambda: self.similaritySort.emit(self, True))
        # Only greyed out if this tile has neither label at all -- main_window's
        # handler picks whichever of new_label/original_label is actually set
        # (preferring new_label) to filter by, so either one being present is
        # enough for this to do something.
        similarity_sort_same_label.setDisabled(not (self.row.label or self.row.original_label))

        menu.exec(event.screenPos())


__all__ = ["RectWidget", "THUMB_SIZE"]
