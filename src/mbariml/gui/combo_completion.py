"""Inline label completion for an editable QComboBox.

Qt's own completer (the default on an editable combo box) stopped completing
partway through review sessions until a restart. It is wired to the line edit
only between a focus-in and a focus-out event, and it is fed only by key
presses -- text that arrives through the macOS input method (an accent popup,
dictation) never reaches it. This completes from ``textEdited`` instead,
which every user edit emits however it arrived and whatever focus did.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QComboBox

# Keys after which the edit was a deletion: filling the text straight back in
# would make Backspace over a completion do nothing. Any key with Cmd/Ctrl
# (cut, undo) counts too.
_DELETE_KEYS = {Qt.Key.Key_Backspace, Qt.Key.Key_Delete}


class _InlineCompleter(QObject):
    def __init__(self, combo: QComboBox) -> None:
        super().__init__(combo)
        self._combo = combo
        self._line_edit = combo.lineEdit()
        self._deleting = False
        # Both: the combo has the focus and hands key presses to its line edit
        # directly, which an event filter on the line edit alone never sees.
        combo.installEventFilter(self)
        self._line_edit.installEventFilter(self)
        self._line_edit.textEdited.connect(self._on_edited)

    def eventFilter(self, source, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress:
            self._deleting = event.key() in _DELETE_KEYS or bool(
                event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
            )
        return False

    def _on_edited(self, text: str) -> None:
        deleting, self._deleting = self._deleting, False
        line_edit = self._line_edit
        if not text or deleting or line_edit.cursorPosition() != len(text):
            return
        prefix = text.casefold()
        for i in range(self._combo.count()):
            item = self._combo.itemText(i)
            if len(item) > len(text) and item.casefold().startswith(prefix):
                line_edit.setText(item)
                line_edit.setSelection(len(text), len(item) - len(text))
                return


def complete_inline(combo: QComboBox) -> None:
    """Typing the start of an item fills in the rest, selected, so typing on
    replaces it and Backspace removes it; case-insensitive, first match in
    item order, the item's own spelling. Replaces the combo's QCompleter."""
    combo.setCompleter(None)
    _InlineCompleter(combo)  # parented to the combo, which keeps it alive


__all__ = ["complete_inline"]
