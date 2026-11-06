"""The review GUI's "Sort..." dialog: a stack of sort keys.

Modelled on vars-gridview's sort dialog (``ui/dialogs/sort_dialog.py``, MIT
License; see ``THIRD_PARTY_NOTICES.md``): a list of keys you Add and drag
into order, where the first key sorts and each next one breaks the ties of
the keys above it -- New Label, then Sharpness gives each label's sharpest
crops first. Unlike vars-gridview, each key also has a direction (Reverse,
or double-click). Keys are ``query_service.SORT_KEYS``; the result is
registered with ``query_service.stacked_sort_option`` and appears in the
Sort by dropdown under its name.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from mbariml.gui import query_service

_KEY_ROLE = Qt.ItemDataRole.UserRole  # (key, descending)


class SortDialog(QDialog):
    """Compose a stacked sort. After ``exec()`` returns Accepted, ``keys()``
    is the stack as [(SORT_KEYS name, descending), ...] -- empty if cleared."""

    def __init__(self, parent=None, keys: list[tuple[str, bool]] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sort")
        self.setMinimumWidth(420)

        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.itemDoubleClicked.connect(self._reverse_item)
        for key, descending in keys or []:
            self._append(key, descending)

        add_button = QPushButton("Add...")
        add_button.clicked.connect(self._add)
        reverse_button = QPushButton("Reverse")
        reverse_button.setToolTip("Flip the selected key's direction (or double-click it).")
        reverse_button.clicked.connect(lambda: self._reverse_item(self._list.currentItem()))
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._remove)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._list.clear)
        buttons_row = QHBoxLayout()
        for button in (add_button, reverse_button, remove_button, clear_button):
            buttons_row.addWidget(button)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Sorted by the first key; each key below breaks the ties of the ones above it.\n"
            "Drag to reorder. E.g. New Label, then Sharpness: each label's sharpest crops first."
        ))
        layout.addWidget(self._list)
        layout.addLayout(buttons_row)
        layout.addWidget(box)

    def keys(self) -> list[tuple[str, bool]]:
        return [self._list.item(i).data(_KEY_ROLE) for i in range(self._list.count())]

    def _append(self, key: str, descending: bool) -> None:
        item = QListWidgetItem(query_service.sort_key_text(key, descending))
        item.setData(_KEY_ROLE, (key, descending))
        self._list.addItem(item)
        self._list.setCurrentItem(item)

    def _add(self) -> None:
        used = {key for key, _ in self.keys()}
        remaining = [key for key in query_service.SORT_KEYS if key not in used]
        if not remaining:
            return
        key, ok = QInputDialog.getItem(self, "Add sort key", "Then sort by:", remaining, 0, False)
        if ok:
            self._append(key, query_service.SORT_KEYS[key][1])

    def _reverse_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        key, descending = item.data(_KEY_ROLE)
        item.setData(_KEY_ROLE, (key, not descending))
        item.setText(query_service.sort_key_text(key, not descending))

    def _remove(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)
