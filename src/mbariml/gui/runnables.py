"""Qt thread-pool worker utilities.

Adapted from MBARI vars-gridview's ``lib/runtime/runnables.py`` (MIT
License; see ``THIRD_PARTY_NOTICES.md``), translated from PyQt6 to PySide6.
Thin wrapper around :class:`~PySide6.QtCore.QRunnable` for dispatching
off-thread work and reporting results back to the GUI thread via signals.

Typical usage::

    worker = Worker(load_payload, arg1, arg2)
    worker.signals.result.connect(on_result)
    worker.signals.error.connect(on_error)
    enqueue(worker)
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import TypeVar

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


def enqueue(runnable: QRunnable) -> None:
    """Submit *runnable* to the global :class:`~PySide6.QtCore.QThreadPool`."""
    QThreadPool.globalInstance().start(runnable)


class WorkerSignals(QObject):
    """Signals emitted by a :class:`Worker` runnable.

    Attributes:
        finished: Emitted when the worker completes (success or error).
        error: Emitted with ``(exc_type, exc_value, traceback_str)`` on failure.
        result: Emitted with the return value on success.
    """

    finished = Signal()
    error = Signal(tuple)  # (type, Exception, str)
    result = Signal(object)


T = TypeVar("T")


class Worker(QRunnable):
    """Generic :class:`~PySide6.QtCore.QRunnable` that wraps a callable.

    Args:
        fn: The callable to run on a pool thread.
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Signals:
        signals.result: Emitted with the return value of *fn*.
        signals.error: Emitted as ``(type, exc, traceback_str)`` on exception.
        signals.finished: Always emitted when the run completes.
    """

    def __init__(self, fn: Callable[..., T], *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Execute the wrapped callable on the pool thread."""
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self.signals.error.emit((type(exc), exc, tb))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


__all__ = ["Worker", "WorkerSignals", "enqueue"]
