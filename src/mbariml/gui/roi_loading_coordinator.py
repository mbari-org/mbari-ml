"""Coordinator for asynchronous ROI tile loading lifecycle in the mosaic.

Adapted from MBARI vars-gridview's
``ui/coordinators/mosaic_roi_loading_coordinator.py`` (MIT License; see
``THIRD_PARTY_NOTICES.md``), translated from PyQt6 to PySide6. The original
also ran a delayed "sweep" retry pass (a few automatic re-attempts at
reduced concurrency, seconds apart) for tiles that failed to load, because
its ROI pixels come from HTTP microservices that can transiently 503 under
load. That doesn't apply here -- ``RectWidget.get_roi()`` just decodes a
JPEG blob already sitting in the query result, which doesn't fail
transiently -- so the sweep-retry machinery is dropped; a tile that fails to
decode just shows its placeholder.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QObject, Signal, Slot

from mbariml.logging_utils import get_logger

if TYPE_CHECKING:
    from mbariml.gui.rect_widget import RectWidget

logger = get_logger(__name__)


class MosaicRoiLoadingCoordinator(QObject):
    """Own ROI loading queueing and completion callback.

    Does not own any UI presentation itself; callers connect to
    :attr:`progress` to drive their own shared progress display.
    """

    progress = Signal(int, int)

    def __init__(self, *, parent: QObject, max_concurrency: int = 8) -> None:
        super().__init__(parent)
        self._max_concurrency = max(1, int(max_concurrency))

        self._generation = 0
        self._total = 0
        self._done = 0
        self._pending: list[RectWidget] = []
        self._inflight = 0
        self._pass_finished = True
        self._top_level_on_complete: Callable[[], None] | None = None
        self._cancel_event: Event | None = None

    def cancel_pending(self) -> None:
        """Invalidate in-flight ROI refreshes."""
        self._generation += 1
        self._total = 0
        self._done = 0
        self._pending = []
        self._inflight = 0
        self._pass_finished = True
        self._top_level_on_complete = None
        self._cancel_event = None

    def start_loading(
        self,
        *,
        rect_widgets: list[RectWidget],
        on_complete: Callable[[], None],
        cancel_event: Event,
    ) -> None:
        """Start batched async ROI loading for the given widgets."""
        if not rect_widgets:
            on_complete()
            return

        self.cancel_pending()
        self._top_level_on_complete = on_complete
        self._cancel_event = cancel_event

        self._generation += 1
        generation = self._generation

        self._total = len(rect_widgets)
        self._done = 0
        self._pending = list(rect_widgets)
        self._inflight = 0
        self._pass_finished = False

        self.progress.emit(self._done, self._total)
        self._pump(generation)

    def _pump(self, generation: int) -> None:
        while (
            generation == self._generation
            and self._inflight < self._max_concurrency
            and self._pending
            and not (self._cancel_event is not None and self._cancel_event.is_set())
        ):
            rect_widget = self._pending.pop(0)
            rect_widget.assign_roi_batch_generation(generation)
            rect_widget.roiRefreshed.connect(self._on_rect_roi_refreshed)
            self._inflight += 1
            rect_widget.request_roi_refresh()

    @Slot(object)
    def _on_rect_roi_refreshed(self, rect_widget: object) -> None:
        rw = cast("RectWidget", rect_widget)
        try:
            rw.roiRefreshed.disconnect(self._on_rect_roi_refreshed)
        except (RuntimeError, TypeError) as exc:
            logger.debug("roiRefreshed already disconnected: %s", exc)

        if rw.roi_batch_generation != self._generation:
            return

        self._inflight = max(0, self._inflight - 1)
        self._done += 1
        self.progress.emit(self._done, self._total)

        self._pump(self._generation)

        cancelled = self._cancel_event is not None and self._cancel_event.is_set()
        if cancelled:
            if self._inflight == 0:
                self._top_level_on_complete = None
            return

        if self._done >= self._total:
            self._finish_pass()

    def _finish_pass(self) -> None:
        # _pump's recursion means every stack frame between the first tile's
        # completion and the last one re-checks `done >= total` once control
        # unwinds back to it -- guard so a pass only finishes once.
        if self._pass_finished:
            return
        self._pass_finished = True

        callback = self._top_level_on_complete
        if callback is not None:
            callback()


__all__ = ["MosaicRoiLoadingCoordinator"]
