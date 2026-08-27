"""Graphics-scene manager for the ROI mosaic grid.

Adapted from MBARI vars-gridview's ``ui/mosaic/mosaic_view.py`` (MIT
License; see ``THIRD_PARTY_NOTICES.md``), translated from PyQt6 to PySide6.
This module has no VARS-specific logic at all -- it's a generic
``QGraphicsScene``/``QGraphicsGridLayout`` renderer that only cares about
widget geometry, so it ports essentially unchanged.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

MOSAIC_LAYOUT_MARGINS = (0, 0, 0, 0)
MOSAIC_LAYOUT_SPACING = 0


@dataclass
class MosaicRenderResult:
    """Summary of a mosaic render pass."""

    columns: int
    rendered_count: int


@dataclass
class MosaicVisibilityFilters:
    """Visibility toggles used to choose which widgets are rendered."""

    hide_verified: bool = False


class MosaicView:
    """Owns QGraphicsScene/QGraphicsWidget layout for mosaic tiles."""

    def __init__(self, graphics_view: QtWidgets.QGraphicsView) -> None:
        self._graphics_view = graphics_view
        self._graphics_scene = QtWidgets.QGraphicsScene()
        self._graphics_widget = QtWidgets.QGraphicsWidget()
        self._layout = QtWidgets.QGraphicsGridLayout()
        self._last_visible_signature: tuple[int, ...] = ()
        self._last_columns: int = -1
        self._last_widget_size: tuple[int, int] = (-1, -1)
        self._init_graphics()

    @property
    def graphics_view(self) -> QtWidgets.QGraphicsView:
        return self._graphics_view

    def _init_graphics(self) -> None:
        self._graphics_view.setScene(self._graphics_scene)
        # A Qt stylesheet's background-color on the QGraphicsView only
        # reaches the viewport widget, not what QGraphicsScene actually
        # paints beneath the tiles -- that needs its own brush, or the
        # scene renders its default (light gray/white) regardless of the
        # rest of the window's dark theme.
        self._graphics_scene.setBackgroundBrush(QtGui.QColor("#1e1f22"))
        self._graphics_scene.addItem(self._graphics_widget)
        self._layout.setContentsMargins(*MOSAIC_LAYOUT_MARGINS)
        self._layout.setHorizontalSpacing(MOSAIC_LAYOUT_SPACING)
        self._layout.setVerticalSpacing(MOSAIC_LAYOUT_SPACING)
        self._graphics_widget.setLayout(self._layout)

    def _clear_graphics_layout(self) -> None:
        while self._layout.count() > 0:
            self._layout.removeAt(0)

    def clear(self, rect_widgets: list) -> None:
        """Hide all widgets, detach them from the scene graph, clear the
        layout, and release each widget's reference cycle back to its
        owning window (see RectWidget.cleanup's docstring) -- all needed for
        a retired tile to be collected immediately and deterministically
        instead of relying on an unpredictable later GC pass, confirmed via
        an actual SIGSEGV to be a real hazard here, not just a delay.

        ``self._layout.removeAt()`` alone is NOT enough: it drops the
        widget from the layout's own position-tracking, but Qt's grid
        layout still holds it as a child of ``self._graphics_widget``
        (established when it was added via ``addItem()``) -- confirmed
        directly: a widget removed only via ``removeAt()`` was still not
        collectible even after an explicit ``gc.collect()``.
        ``setParentItem(None)`` is what actually releases that scene-graph
        hold.
        """
        for rect_widget in rect_widgets:
            rect_widget.hide()
            rect_widget.setParentItem(None)
            cleanup = getattr(rect_widget, "cleanup", None)
            if cleanup is not None:
                cleanup()

        self._clear_graphics_layout()
        self._graphics_scene.setSceneRect(QtCore.QRectF())
        self._last_visible_signature = ()
        self._last_columns = -1
        self._last_widget_size = (-1, -1)

        # Defense in depth: force any remaining cycle among the widgets just
        # detached above (ours or a third-party one we haven't found) to be
        # collected right here -- a moment we control -- rather than by an
        # unpredictable later GC pass, which for a QGraphicsItem is a real
        # crash hazard (see this method's docstring). Callers build the next
        # batch of tiles immediately after this returns, so this is also
        # exactly the safe place for it: not nested inside construction of
        # anything else.
        gc.collect()

    def render(
        self,
        *,
        all_widgets: list,
        visible_widgets: list,
    ) -> MosaicRenderResult:
        """Render `visible_widgets` in a responsive grid."""
        left, _top, right, _bottom = self._layout.getContentsMargins()
        left_i = self._coerce_margin(left)
        right_i = self._coerce_margin(right)
        viewport = self._graphics_view.viewport()
        viewport_width = viewport.width() if viewport is not None else 0
        width = viewport_width - left_i - right_i

        if all_widgets:
            rect_widget_width_f = all_widgets[0].boundingRect().width()
            rect_widget_height_f = all_widgets[0].boundingRect().height()

            # Defensive guard: stale/invalid geometry can transiently report zero.
            rect_widget_width = max(round(rect_widget_width_f), 1)
            rect_widget_height = max(round(rect_widget_height_f), 1)

            columns = max(int(width / rect_widget_width), 1)
        else:
            rect_widget_width = 0
            rect_widget_height = 0
            columns = 1

        visible_signature = tuple(id(widget) for widget in visible_widgets)
        if (
            visible_signature == self._last_visible_signature
            and columns == self._last_columns
            and (rect_widget_width, rect_widget_height) == self._last_widget_size
        ):
            return MosaicRenderResult(
                columns=columns, rendered_count=len(visible_widgets)
            )

        self._graphics_view.setUpdatesEnabled(False)
        try:
            self._clear_graphics_layout()

            visible_ids = set(visible_signature)
            for rw in all_widgets:
                if id(rw) not in visible_ids:
                    rw.hide()

            for idx, rect_widget in enumerate(visible_widgets):
                row = idx // columns
                col = idx % columns
                self._layout.addItem(rect_widget, row, col)
                rect_widget.show()

            rows = (len(visible_widgets) + columns - 1) // columns
            self._graphics_widget.resize(
                columns * rect_widget_width,
                rows * rect_widget_height,
            )
            self._graphics_scene.setSceneRect(self._graphics_widget.boundingRect())
        finally:
            self._graphics_view.setUpdatesEnabled(True)

        self._last_visible_signature = visible_signature
        self._last_columns = columns
        self._last_widget_size = (rect_widget_width, rect_widget_height)

        return MosaicRenderResult(columns=columns, rendered_count=len(visible_widgets))

    def select_visible_widgets(
        self,
        *,
        all_widgets: list,
        filters: MosaicVisibilityFilters,
    ) -> list:
        """Return widgets that pass current visibility filter toggles."""
        visible_widgets = []
        for widget in all_widgets:
            row = getattr(widget, "row", None)
            if row is None:
                continue
            if filters.hide_verified and getattr(row, "verified", False):
                continue
            visible_widgets.append(widget)
        return visible_widgets

    def ensure_widget_visible_if_needed(self, rect_widget: object) -> None:
        """Scroll view only when `rect_widget` is outside current viewport."""
        viewport = self._graphics_view.viewport()
        if viewport is None:
            return

        item_rect_scene = rect_widget.sceneBoundingRect()
        item_rect_view = self._graphics_view.mapFromScene(
            item_rect_scene
        ).boundingRect()
        viewport_rect = viewport.rect()
        if viewport_rect.contains(item_rect_view):
            return

        self._graphics_view.ensureVisible(item_rect_scene, 8, 8)

    def visible_widgets_in_range(
        self,
        *,
        all_widgets: list,
        begin_index: int,
        end_index: int,
    ) -> list:
        """Return currently visible widgets within an inclusive index range."""
        if begin_index < 0 or end_index < begin_index:
            return []
        max_index = len(all_widgets) - 1
        if max_index < 0:
            return []
        bounded_begin = min(begin_index, max_index)
        bounded_end = min(end_index, max_index)

        visible: list = []
        for idx in range(bounded_begin, bounded_end + 1):
            widget = all_widgets[idx]
            if widget.isVisible():
                visible.append(widget)
        return visible

    @staticmethod
    def compute_relative_index(
        *,
        current_index: int,
        key: QtCore.Qt.Key,
        columns: int,
        total_items: int,
    ) -> int | None:
        """Compute next selection index for arrow-key navigation."""
        if current_index < 0 or current_index >= total_items:
            return None
        if columns <= 0:
            return None

        if key == QtCore.Qt.Key.Key_Left:
            next_index = current_index - 1
        elif key == QtCore.Qt.Key.Key_Right:
            next_index = current_index + 1
        elif key == QtCore.Qt.Key.Key_Up:
            next_index = current_index - columns
        elif key == QtCore.Qt.Key.Key_Down:
            next_index = current_index + columns
        else:
            return None

        if 0 <= next_index < total_items:
            return next_index
        return None

    @staticmethod
    def _coerce_margin(value: object) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        return 0


__all__ = ["MosaicRenderResult", "MosaicView", "MosaicVisibilityFilters"]
