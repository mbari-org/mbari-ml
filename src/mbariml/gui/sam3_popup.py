"""The small popup "Add ROI with SAM3" shows at a click: pick the box, name it, add it.

SAM3 offers up to ~3 nested boxes for one click (a part, the object, the
object plus its surroundings). The popup starts on SAM3's highest-scoring
one; Tab / Shift+Tab (or the arrow buttons) step through the rest, each
previewed as a dashed box on the image while the popup is open. Enter adds
the box with the typed label; Esc, or clicking anywhere outside, cancels
without writing anything.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mbariml.gui.sam3_service import Box, Candidate


class Sam3BoxPopup(QFrame):
    def __init__(
        self,
        parent: QWidget,
        candidates: list[Candidate],
        labels: list[str],
        default_label: str | None,
        *,
        on_preview: Callable[[Box], None],
        on_accept: Callable[[Box, str], None],
        on_cancel: Callable[[], None],
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setFrameShape(QFrame.Shape.Box)
        self._candidates = candidates
        self._index = max(range(len(candidates)), key=lambda i: candidates[i].score)
        self._on_preview, self._on_accept, self._on_cancel = on_preview, on_accept, on_cancel
        self._done = False

        prev_button = QPushButton("◀")
        next_button = QPushButton("▶")
        for button in (prev_button, next_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedWidth(32)
        prev_button.clicked.connect(lambda: self._step(-1))
        next_button.clicked.connect(lambda: self._step(1))
        self._size_label = QLabel()
        size_row = QHBoxLayout()
        size_row.addWidget(prev_button)
        size_row.addWidget(self._size_label, stretch=1)
        size_row.addWidget(next_button)

        self._label_combo = QComboBox()
        self._label_combo.setEditable(True)
        self._label_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._label_combo.addItems(labels)
        self._label_combo.setCurrentText(default_label or "")
        self._label_combo.setMinimumWidth(220)
        self._label_combo.lineEdit().setPlaceholderText("label")
        self._label_combo.lineEdit().returnPressed.connect(self._accept)

        self._hint = QLabel("Tab: next size · Enter: add · Esc: cancel")
        self._hint.setStyleSheet("color: #9a9da3; font-size: 11px;")

        add_button = QPushButton("Add")
        add_button.setAutoDefault(False)
        add_button.setStyleSheet("QPushButton { background-color: #2fa84f; color: white; font-weight: bold; }")
        add_button.clicked.connect(self._accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.setAutoDefault(False)
        cancel_button.clicked.connect(self.close)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(cancel_button)
        button_row.addWidget(add_button)

        layout = QVBoxLayout(self)
        layout.addLayout(size_row)
        layout.addWidget(self._label_combo)
        layout.addWidget(self._hint)
        layout.addLayout(button_row)
        self._show_size()

    def popup_at(self, screen_pos: QPoint) -> None:
        """Show just below-right of the click, so it doesn't cover the object,
        kept on the screen."""
        self.adjustSize()
        screen = QApplication.screenAt(screen_pos) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        x = min(screen_pos.x() + 24, area.right() - self.width())
        y = screen_pos.y() + 24
        if y + self.height() > area.bottom():
            y = screen_pos.y() - 24 - self.height()  # flip above the click
        self.move(max(area.left(), x), max(area.top(), y))
        self.show()
        self._label_combo.setFocus()
        self._label_combo.lineEdit().selectAll()

    def _show_size(self) -> None:
        c = self._candidates[self._index]
        w, h = c.box[2] - c.box[0], c.box[3] - c.box[1]
        self._size_label.setText(
            f"Box {self._index + 1} of {len(self._candidates)} — {w:.0f}×{h:.0f} px, score {c.score:.2f}"
        )
        self._on_preview(c.box)

    def _step(self, delta: int) -> None:
        self._index = (self._index + delta) % len(self._candidates)
        self._show_size()

    def focusNextPrevChild(self, next_: bool) -> bool:  # noqa: N802 (Qt's own naming)
        # Tab / Shift+Tab step through SAM3's boxes instead of moving focus.
        self._step(1 if next_ else -1)
        return True

    def _accept(self) -> None:
        label = self._label_combo.currentText().strip()
        if not label:
            self._hint.setText("Type a label first (or Esc to cancel).")
            return
        self._done = True
        try:
            self._on_accept(self._candidates[self._index].box, label)
        finally:
            self.close()  # even if saving failed: a stuck popup would save twice on the next Enter

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's own naming)
        if not self._done:
            self._done = True
            self._on_cancel()
        super().closeEvent(event)


__all__ = ["Sam3BoxPopup"]
