from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class DropdownSelect(QWidget):
    currentTextChanged = Signal(str)
    currentIndexChanged = Signal(int)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        max_popup_height: int = 220,
        popup_direction: str = "auto",
    ) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, Any]] = []
        self._current_index = -1
        self._is_open = False
        self._opens_up = False
        self._max_popup_height = max_popup_height
        self._popup_direction = popup_direction
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(36)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QFrame()
        self.header.setObjectName("dropdownHeader")
        self.header.setProperty("open", False)
        self.header.setProperty("opensUp", False)
        self.header.setCursor(Qt.PointingHandCursor)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(10, 0, 8, 0)
        header_layout.setSpacing(8)

        self.label = QLabel("")
        self.label.setObjectName("dropdownLabel")
        self.label.setTextInteractionFlags(Qt.NoTextInteraction)
        header_layout.addWidget(self.label, 1)

        self.arrow = QLabel("v")
        self.arrow.setObjectName("dropdownArrow")
        self.arrow.setAlignment(Qt.AlignCenter)
        self.arrow.setFixedWidth(18)
        header_layout.addWidget(self.arrow, 0)
        layout.addWidget(self.header)

        self.popup = QFrame(None, Qt.Popup | Qt.FramelessWindowHint)
        self.popup.setObjectName("dropdownPopup")
        self.popup.setProperty("opensUp", False)
        self.popup.setAttribute(Qt.WA_StyledBackground, True)
        self.popup.installEventFilter(self)
        popup_layout = QVBoxLayout(self.popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        popup_layout.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("dropdownScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.option_host = QWidget()
        self.option_host.setObjectName("dropdownOptionHost")
        self.panel_layout = QVBoxLayout(self.option_host)
        self.panel_layout.setContentsMargins(4, 4, 4, 4)
        self.panel_layout.setSpacing(2)
        self.scroll.setWidget(self.option_host)
        popup_layout.addWidget(self.scroll)

        self.header.mousePressEvent = self._header_mouse_press  # type: ignore[method-assign]

    def eventFilter(self, watched, event) -> bool:
        if watched is self.popup and event.type() == QEvent.Hide and self._is_open:
            self._is_open = False
            self._apply_open_state(False, False)
        return super().eventFilter(watched, event)

    def addItem(self, label: str, data: Any = None) -> None:
        self._items.append((label, data))
        self._rebuild_options()
        if self._current_index < 0:
            self.setCurrentIndex(0)

    def addItems(self, labels: list[str] | tuple[str, ...]) -> None:
        for label in labels:
            self.addItem(str(label), str(label))

    def clear(self) -> None:
        self._items.clear()
        self._current_index = -1
        self._rebuild_options()
        self.label.setText("")
        self.hidePopup()

    def currentText(self) -> str:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][0]
        return ""

    def currentData(self) -> Any:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def setCurrentText(self, text: str) -> None:
        for index, (label, _data) in enumerate(self._items):
            if label == text:
                self.setCurrentIndex(index)
                return

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items):
            return
        changed = index != self._current_index
        self._current_index = index
        self.label.setText(self._items[index][0])
        self._rebuild_options()
        if changed:
            self.currentIndexChanged.emit(index)
            self.currentTextChanged.emit(self.currentText())

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.header.setEnabled(enabled)
        if not enabled:
            self.hidePopup()

    def hidePopup(self) -> None:
        if not self._is_open:
            return
        self._is_open = False
        self.popup.hide()
        self._apply_open_state(False, False)

    def showPopup(self) -> None:
        if not self._items or not self.isEnabled():
            return
        self._rebuild_options()
        screen = self.screen() or QApplication.primaryScreen()
        screen_rect = screen.availableGeometry() if screen is not None else self.window().geometry()
        global_pos = self.mapToGlobal(QPoint(0, 0))
        below_space = screen_rect.bottom() - (global_pos.y() + self.height())
        above_space = global_pos.y() - screen_rect.top()
        desired_height = min(self._content_height(), self._max_popup_height)
        if self._popup_direction == "up":
            opens_up = True
        elif self._popup_direction == "down":
            opens_up = False
        else:
            opens_up = below_space < desired_height and above_space > below_space
        available_height = above_space if opens_up else below_space
        popup_height = max(42, min(desired_height, max(42, available_height - 4)))
        popup_width = self.width()
        x = global_pos.x()
        y = global_pos.y() - popup_height if opens_up else global_pos.y() + self.height()
        self.scroll.setMaximumHeight(popup_height)
        self.popup.resize(popup_width, popup_height)
        self.popup.move(x, y)
        self._is_open = True
        self._opens_up = opens_up
        self._apply_open_state(True, opens_up)
        self.popup.show()
        self.popup.raise_()
        self.popup.activateWindow()

    def _header_mouse_press(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self._is_open:
                self.hidePopup()
            else:
                self.showPopup()
            event.accept()
            return
        event.ignore()

    def _apply_open_state(self, open_: bool, opens_up: bool) -> None:
        self.header.setProperty("open", open_)
        self.header.setProperty("opensUp", opens_up)
        self.popup.setProperty("opensUp", opens_up)
        for widget in (self.header, self.popup):
            widget.style().unpolish(widget)
            widget.style().polish(widget)
        if open_:
            self.arrow.setText("v" if opens_up else "^")
        else:
            self.arrow.setText("^" if self._popup_direction == "up" else "v")

    def _select_index(self, index: int) -> None:
        self.setCurrentIndex(index)
        self.hidePopup()

    def _content_height(self) -> int:
        row_height = 34
        return max(42, len(self._items) * row_height + 8)

    def _rebuild_options(self) -> None:
        while self.panel_layout.count():
            item = self.panel_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, (label, _data) in enumerate(self._items):
            button = QPushButton(label)
            button.setObjectName("dropdownOption")
            button.setProperty("selected", index == self._current_index)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(30)
            button.clicked.connect(lambda checked=False, idx=index: self._select_index(idx))
            self.panel_layout.addWidget(button)
        self.panel_layout.addStretch(1)
