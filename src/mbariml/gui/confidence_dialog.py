"""Per-class table of what the review GUI's confidence range hides.

Opened from the "Details..." button beside the confidence slider. The rows
come from ``query_service.count_hidden_by_confidence``; nothing here touches
the database.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

_HEADERS = ("Class", "Total", "Shown", "Hidden", "% hidden")


def summarize(rows: list[tuple[str, int, int]]) -> tuple[int, int, int, int]:
    """(total ROIs, hidden ROIs, classes with any hidden, classes) for *rows*."""
    total = sum(r[1] for r in rows)
    hidden = sum(r[2] for r in rows)
    return total, hidden, sum(1 for r in rows if r[2]), len(rows)


class _NumberItem(QTableWidgetItem):
    """Shows formatted text ("12,611", "41.0%") but sorts by the number
    behind it, so a column sort is numeric (9 before 10), not alphabetical."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        return self.data(Qt.ItemDataRole.UserRole) < other.data(Qt.ItemDataRole.UserRole)


def _number_item(value: float, text: str) -> QTableWidgetItem:
    item = _NumberItem(text)
    item.setData(Qt.ItemDataRole.UserRole, value)
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class ConfidenceHiddenDialog(QDialog):
    def __init__(
        self,
        rows: list[tuple[str, int, int]],
        *,
        min_confidence: float,
        max_confidence: float,
        label_filter: str | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hidden by the confidence range")
        self.resize(620, 640)
        self._rows = rows

        total, hidden, classes_hit, classes = summarize(rows)
        fully_hidden = sum(1 for r in rows if r[1] and r[1] == r[2])
        scope = f"search “{label_filter}”" if label_filter else "the whole database"
        share = f" ({hidden / total:.1%})" if total else ""
        summary = QLabel(
            f"Confidence {min_confidence:.2f}-{max_confidence:.2f} hides <b>{hidden:,}</b> of {total:,} ROIs"
            f"{share} in {scope}: {classes_hit} of {classes} classes lose ROIs, {fully_hidden} entirely. "
            "Counts ignore Hide verified/unverified. Click a column header to sort."
        )
        summary.setWordWrap(True)

        table = QTableWidget(len(rows), len(_HEADERS))
        table.setHorizontalHeaderLabels(_HEADERS)
        table.verticalHeader().setVisible(False)
        for i, (label, n, n_hidden) in enumerate(rows):
            name = QTableWidgetItem(label)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, 0, name)
            table.setItem(i, 1, _number_item(n, f"{n:,}"))
            table.setItem(i, 2, _number_item(n - n_hidden, f"{n - n_hidden:,}"))
            table.setItem(i, 3, _number_item(n_hidden, f"{n_hidden:,}"))
            pct = round(100 * n_hidden / n, 1) if n else 0.0
            table.setItem(i, 4, _number_item(pct, f"{pct:.1f}%"))
        table.setSortingEnabled(True)
        table.sortItems(3, Qt.SortOrder.DescendingOrder)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

        copy_button = QPushButton("Copy table")
        copy_button.setToolTip("Copy as tab-separated text, ready to paste into a spreadsheet.")
        copy_button.clicked.connect(self._copy)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton(copy_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(table)
        layout.addWidget(buttons)

    def _copy(self) -> None:
        lines = ["\t".join(_HEADERS)]
        for label, n, n_hidden in self._rows:
            pct = 100 * n_hidden / n if n else 0.0
            lines.append(f"{label}\t{n}\t{n - n_hidden}\t{n_hidden}\t{pct:.1f}")
        QGuiApplication.clipboard().setText("\n".join(lines) + "\n")
