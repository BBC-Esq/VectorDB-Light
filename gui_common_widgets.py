# gui_common_widgets.py
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox
from typing import Callable, Optional


class RefreshingComboBox(QComboBox):

    def __init__(
        self,
        parent=None,
        get_items: Optional[Callable[[], list[str]]] = None,
        placeholder: Optional[str] = None,
    ):
        super().__init__(parent)
        self._get_items = get_items
        self._placeholder = placeholder

        if placeholder:
            self._add_placeholder()

    def _add_placeholder(self):
        self.addItem(self._placeholder)
        self.setItemData(0, QColor('gray'), Qt.ForegroundRole)

    def set_item_source(self, get_items: Callable[[], list[str]]):
        self._get_items = get_items

    def refresh_items(self):
        if self._get_items is None:
            return

        current_text = self.currentText()
        is_placeholder = self._placeholder and current_text == self._placeholder

        self.blockSignals(True)
        self.clear()

        if self._placeholder:
            self._add_placeholder()

        items = self._get_items()
        self.addItems(items)

        if not is_placeholder and current_text:
            index = self.findText(current_text)
            if index >= 0:
                self.setCurrentIndex(index)
                self.blockSignals(False)
                return

        self.setCurrentIndex(0)
        self.blockSignals(False)

    def showPopup(self):
        if self._get_items is not None:
            self.refresh_items()
        super().showPopup()