"""Selection state and keyboard-navigation logic for the mosaic grid.

Adapted from MBARI vars-gridview's ``controllers/selection_model.py`` and
``ui/coordinators/mosaic_selection_coordinator.py`` (MIT License; see
``THIRD_PARTY_NOTICES.md``), translated from PyQt6 to PySide6 and merged
into one module. Neither of these had any VARS-specific logic -- they only
deal in generic tile widgets and index arithmetic -- so the port is close
to unchanged, aside from combining the two classes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from PySide6 import QtCore
from PySide6.QtCore import QObject, Signal

from mbariml.gui.mosaic_view import MosaicView

if TYPE_CHECKING:
    from mbariml.gui.rect_widget import RectWidget


class SelectionModel(QObject):
    """Central store for the current tile selection.

    Maintains an ordered list of selected tile widgets and emits
    :attr:`selection_changed` whenever the selection is modified.
    """

    selection_changed = Signal(list)  # list[RectWidget]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._selected: list[RectWidget] = []

    @property
    def selected(self) -> list[RectWidget]:
        """Snapshot of the currently selected widgets (read-only)."""
        return list(self._selected)

    @property
    def count(self) -> int:
        return len(self._selected)

    def is_selected(self, widget: RectWidget) -> bool:
        return widget in self._selected

    def set_selection(self, widgets: list[RectWidget]) -> None:
        if self._selected == widgets:
            return
        self._selected = list(widgets)
        self.selection_changed.emit(self._selected)

    def add(self, widget: RectWidget) -> None:
        if widget not in self._selected:
            self._selected.append(widget)
            self.selection_changed.emit(self._selected)

    def remove(self, widget: RectWidget) -> None:
        if widget in self._selected:
            self._selected.remove(widget)
            self.selection_changed.emit(self._selected)

    def toggle(self, widget: RectWidget) -> None:
        if widget in self._selected:
            self.remove(widget)
        else:
            self.add(widget)

    def clear(self) -> None:
        if self._selected:
            self._selected = []
            self.selection_changed.emit(self._selected)


class _MosaicViewLike(Protocol):
    def visible_widgets_in_range(
        self,
        *,
        all_widgets: list,
        begin_index: int,
        end_index: int,
    ) -> list: ...

    def ensure_widget_visible_if_needed(self, rect_widget: object) -> None: ...


class MosaicSelectionCoordinator(QObject):
    """Own selection state transitions for mosaic rect widgets."""

    def __init__(
        self,
        *,
        parent: QObject,
        selection_model: SelectionModel,
        mosaic_view: _MosaicViewLike,
        all_widgets_getter: Callable[[], list],
        visible_widgets_getter: Callable[[], list] | None = None,
    ) -> None:
        super().__init__(parent)
        self._selection_model = selection_model
        self._mosaic_view = mosaic_view
        self._all_widgets_getter = all_widgets_getter
        # Keyboard navigation (select_relative) needs the widgets in the
        # order/set actually laid out in the grid right now, not every
        # loaded widget -- when a visibility filter (e.g. "Hide verified")
        # is active, those differ, and index arithmetic done over the full
        # list while `columns` reflects the filtered layout can land on a
        # hidden widget: internally "selected" but never visibly shown,
        # which looks exactly like arrow-key navigation "not following".
        # Falls back to all_widgets_getter when no filtering is in play
        # (e.g. a caller/test that doesn't distinguish the two).
        self._visible_widgets_getter = visible_widgets_getter or all_widgets_getter
        # Anchor: the fixed endpoint for Shift-based range selection.
        # Nav cursor: the moving endpoint (updated by Shift+Arrow and Shift+Click).
        # Both are reset to the clicked/navigated item on any non-Shift operation.
        self._anchor: RectWidget | None = None
        self._nav_cursor: RectWidget | None = None

    @property
    def anchor(self) -> RectWidget | None:
        """The fixed anchor for Shift-based range selection."""
        return self._anchor

    def reset(self) -> None:
        """Clear selection and forget the anchor/nav-cursor widgets entirely.

        Unlike a plain ``selection_model.clear()``, this also drops the
        coordinator's *own* references to specific widgets. Call this before
        discarding the widget list itself (e.g. a page reload) -- otherwise
        a stale anchor or nav-cursor keeps a discarded widget reachable,
        which is exactly the kind of leftover reference that (combined with
        the reference cycle RectWidget.cleanup() breaks) can leave a
        QGraphicsItem waiting on the unpredictable cyclic GC instead of
        being collected immediately.
        """
        self._selection_model.clear()
        self._anchor = None
        self._nav_cursor = None

    def select(self, rect_widget: object, *, clear: bool = True) -> None:
        all_widgets = self._all_widgets_getter()
        if rect_widget not in all_widgets:
            raise ValueError("Widget not in rect widget list")
        rect = cast("RectWidget", rect_widget)

        if clear:
            self._selection_model.set_selection([rect])
        else:
            self._selection_model.add(rect)
        self._anchor = rect
        self._nav_cursor = rect

    def deselect(self, rect_widget: object) -> None:
        all_widgets = self._all_widgets_getter()
        if rect_widget not in all_widgets:
            raise ValueError("Widget not in rect widget list")
        rect = cast("RectWidget", rect_widget)
        self._selection_model.remove(rect)
        self._anchor = rect
        self._nav_cursor = rect

    def select_range(self, first: object, last: object, *, add: bool = False) -> None:
        """Select visible widgets between *first* and *last* (inclusive).

        Args:
            first: Anchor end of the range.
            last: Active end of the range (nav cursor is updated to this widget).
            add: When ``True``, union the range with the existing selection instead
                of replacing it (Ctrl+Shift behaviour).
        """
        all_widgets = self._all_widgets_getter()
        if first not in all_widgets:
            raise ValueError("First widget not in rect widget list")
        if last not in all_widgets:
            raise ValueError("Last widget not in rect widget list")

        first_idx = all_widgets.index(first)
        last_idx = all_widgets.index(last)
        begin_idx = min(first_idx, last_idx)
        end_idx = max(first_idx, last_idx)

        range_selection = list(
            self._mosaic_view.visible_widgets_in_range(
                all_widgets=all_widgets,
                begin_index=begin_idx,
                end_index=end_idx,
            )
        )
        if add:
            existing = self._selection_model.selected
            combined = list(dict.fromkeys(existing + range_selection))
            self._selection_model.set_selection(combined)
        else:
            self._selection_model.set_selection(range_selection)
        # Anchor is intentionally left unchanged; only nav cursor moves.
        self._nav_cursor = cast("RectWidget", last)

    def update_widget_selection_flags(self, selected: list) -> None:
        selected_set = set(selected)
        for rect_widget in self._all_widgets_getter():
            is_selected = rect_widget in selected_set
            if rect_widget.is_selected != is_selected:
                rect_widget.is_selected = is_selected
                rect_widget.update()

    def select_relative(
        self,
        *,
        key: QtCore.Qt.Key,
        columns: int,
        activate_callback: Callable[[object], None],
        shift: bool = False,
    ) -> bool:
        # Deliberately the VISIBLE set (see visible_widgets_getter's
        # docstring in __init__), not every loaded widget -- must match
        # `columns`, which the caller computes from the same rendered
        # layout, or index arithmetic and grid geometry disagree.
        all_widgets = self._visible_widgets_getter()
        navigable_keys = {
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
        }
        if key not in navigable_keys:
            return False

        if shift:
            if self._anchor is None:
                return True
            nav = self._nav_cursor if self._nav_cursor is not None else self._anchor
            if nav not in all_widgets:
                nav = self._anchor
            if nav not in all_widgets:
                return False
            nav_idx = all_widgets.index(nav)
            next_idx = MosaicView.compute_relative_index(
                current_index=nav_idx,
                key=key,
                columns=columns,
                total_items=len(all_widgets),
            )
            if next_idx is None:
                return False
            next_widget = all_widgets[next_idx]
            anchor_idx = all_widgets.index(self._anchor)
            begin_idx = min(anchor_idx, next_idx)
            end_idx = max(anchor_idx, next_idx)
            range_selection = list(
                self._mosaic_view.visible_widgets_in_range(
                    all_widgets=all_widgets,
                    begin_index=begin_idx,
                    end_index=end_idx,
                )
            )
            self._selection_model.set_selection(range_selection)
            self._nav_cursor = next_widget
            self._mosaic_view.ensure_widget_visible_if_needed(next_widget)
            return True

        selected = self._selection_model.selected
        if len(selected) == 0:
            return True

        # Navigate from the nav cursor so arrow keys after a Shift+selection
        # continue from the live end of the range, not from the anchor.
        nav = self._nav_cursor
        if nav is None or nav not in all_widgets:
            nav = selected[0]
        nav_idx = all_widgets.index(nav)
        next_idx = MosaicView.compute_relative_index(
            current_index=nav_idx,
            key=key,
            columns=columns,
            total_items=len(all_widgets),
        )
        if next_idx is None:
            return False

        # No separate clear() before activate_callback(): main_window.py's
        # activate_callback is `self._on_rect_clicked(widget, None)`, which
        # (with no event, i.e. no modifiers) always takes the plain-click
        # branch -- `self._selection.select(rect_widget, clear=True)` --
        # and that already atomically replaces the old selection with the
        # new one via SelectionModel.set_selection(), a single
        # selection_changed emission. An earlier revision called clear()
        # here first, which meant every arrow press did the FULL selection
        # transition TWICE: once to empty (repainting every tile via
        # update_widget_selection_flags, and recoloring every currently-
        # loaded detail-view box via DetailView.update_active), then
        # immediately again to the new widget -- and on a press that also
        # crosses to a different source image, that first "recolor
        # everything inactive" pass was pure waste, immediately followed by
        # a full detail-view box teardown/rebuild that made it moot. Two
        # full passes over up to hundreds of QGraphicsItems per keystroke,
        # for no reason -- doubling exactly the kind of repeated
        # QGraphicsItem churn this codebase has traced to real crashes
        # elsewhere (see RectWidget.cleanup()'s docstring).
        next_widget = all_widgets[next_idx]
        activate_callback(next_widget)
        self._mosaic_view.ensure_widget_visible_if_needed(next_widget)
        return True


__all__ = ["SelectionModel", "MosaicSelectionCoordinator"]
