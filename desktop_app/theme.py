from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f5f7fb"))
    palette.setColor(QPalette.WindowText, QColor("#162033"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#eef2f7"))
    palette.setColor(QPalette.Text, QColor("#162033"))
    palette.setColor(QPalette.Button, QColor("#ffffff"))
    palette.setColor(QPalette.ButtonText, QColor("#162033"))
    palette.setColor(QPalette.Highlight, QColor("#1f9d68"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.PlaceholderText, QColor("#7b8596"))
    app.setPalette(palette)

    app.setStyleSheet(
        """
        QWidget {
            color: #162033;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 13px;
        }
        QMainWindow, QDialog {
            background: #f5f7fb;
        }
        QFrame[card="true"] {
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 8px;
        }
        QFrame[chatBubbleUser="true"] {
            background: #dff3ff;
            border: 1px solid #9fd2ef;
            border-radius: 12px;
        }
        QFrame[chatBubbleAssistant="true"] {
            background: #ffffff;
            border: 1px solid #d8e1ec;
            border-radius: 12px;
        }
        QLabel[chatRole="true"] {
            color: #657084;
            font-size: 12px;
            font-weight: 600;
        }
        QFrame[evidenceCard="true"] {
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 8px;
        }
        QLabel[muted="true"] {
            color: #6b7484;
        }
        QLabel[sectionTitle="true"] {
            font-size: 14px;
            font-weight: 600;
        }
        QLabel[heroTitle="true"] {
            font-size: 20px;
            font-weight: 700;
        }
        QPushButton {
            background: #ffffff;
            border: 1px solid #cfd8e3;
            border-radius: 8px;
            padding: 8px 12px;
        }
        QPushButton:hover {
            border-color: #1f9d68;
        }
        QPushButton:disabled {
            color: #98a1af;
            border-color: #d9e1eb;
        }
        QPushButton[primary="true"] {
            background: #1f9d68;
            color: #ffffff;
            border: 1px solid #1f9d68;
            font-weight: 600;
        }
        QPushButton[secondary="true"] {
            background: #eff7f3;
            border: 1px solid #cfe8dc;
        }
        QFrame#dropdownHeader {
            background: #ffffff;
            border: 1px solid #d7deea;
            border-radius: 8px;
            min-height: 34px;
        }
        QFrame#dropdownHeader:hover {
            border-color: #1f9d68;
        }
        QFrame#dropdownHeader[open="true"] {
            border-color: #1f9d68;
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }
        QFrame#dropdownHeader[open="true"][opensUp="true"] {
            border-color: #1f9d68;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        QLabel#dropdownLabel {
            background: transparent;
            color: #162033;
            padding-left: 0;
        }
        QLabel#dropdownArrow {
            background: transparent;
            color: #526070;
            font-size: 13px;
            font-weight: 700;
        }
        QFrame#dropdownPopup {
            background: #ffffff;
            border: 1px solid #1f9d68;
            border-top: none;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        QFrame#dropdownPopup[opensUp="true"] {
            border: 1px solid #1f9d68;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }
        QScrollArea#dropdownScroll {
            background: transparent;
            border: none;
        }
        QWidget#dropdownOptionHost {
            background: transparent;
        }
        QPushButton#dropdownOption {
            background: transparent;
            border: none;
            border-radius: 6px;
            padding: 7px 8px;
            text-align: left;
        }
        QPushButton#dropdownOption:hover {
            background: #eff7f3;
            border: none;
        }
        QPushButton#dropdownOption[selected="true"] {
            background: #e7f5ee;
            color: #1f9d68;
            font-weight: 600;
        }
        QCheckBox {
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            image: url(desktop_app/assets/check_unchecked.svg);
            border: none;
            background: transparent;
        }
        QCheckBox::indicator:hover {
            image: url(desktop_app/assets/check_unchecked.svg);
            background: transparent;
        }
        QCheckBox::indicator:checked {
            image: url(desktop_app/assets/check_checked.svg);
            background: transparent;
            border: none;
        }
        QCheckBox::indicator:disabled {
            image: url(desktop_app/assets/check_unchecked.svg);
            background: transparent;
        }
        QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QComboBox, QListWidget, QTableWidget, QTreeWidget {
            background: #ffffff;
            border: 1px solid #d7deea;
            border-radius: 8px;
            padding: 6px;
        }
        QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QTextBrowser:focus, QComboBox:focus, QListWidget:focus, QTableWidget:focus, QTreeWidget:focus {
            border: 1px solid #1f9d68;
        }
        QPlainTextEdit, QTextEdit, QTextBrowser {
            selection-background-color: #caead9;
        }
        QComboBox {
            min-height: 22px;
            padding: 6px 34px 6px 10px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 30px;
            border-left: 1px solid #d7deea;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
            background: #f8fafc;
        }
        QComboBox::down-arrow {
            image: url(desktop_app/assets/combo_down.svg);
            width: 14px;
            height: 14px;
        }
        QComboBox QAbstractItemView {
            background: #ffffff;
            border: 1px solid #1f9d68;
            border-radius: 8px;
            padding: 4px;
            outline: 0;
            selection-background-color: #eff7f3;
            selection-color: #162033;
        }
        QComboBox QAbstractItemView::item {
            min-height: 28px;
            padding: 6px 8px;
            border-radius: 6px;
        }
        QTreeWidget::item, QListWidget::item {
            min-height: 24px;
            padding: 4px;
            border-radius: 6px;
        }
        QTreeWidget::item:selected, QListWidget::item:selected {
            background: transparent;
            color: #162033;
        }
        QTreeWidget::indicator {
            width: 20px;
            height: 20px;
            image: url(desktop_app/assets/check_unchecked.svg);
            border: none;
            background: transparent;
        }
        QTreeWidget::indicator:checked {
            image: url(desktop_app/assets/check_checked.svg);
            background: transparent;
            border: none;
        }
        QTreeWidget::indicator:unchecked:hover {
            image: url(desktop_app/assets/check_unchecked.svg);
            background: transparent;
        }
        QTreeWidget::branch {
            background: transparent;
        }
        QTreeWidget::branch:closed:has-children {
            image: url(desktop_app/assets/tree_closed.svg);
        }
        QTreeWidget::branch:open:has-children {
            image: url(desktop_app/assets/tree_open.svg);
        }
        QScrollArea {
            background: transparent;
            border: none;
        }
        QScrollBar:vertical {
            background: #f3f6fa;
            width: 10px;
            margin: 2px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #c7d2df;
            border-radius: 5px;
            min-height: 28px;
        }
        QScrollBar::handle:vertical:hover {
            background: #9fb0c3;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
            border: none;
            background: transparent;
        }
        QScrollBar:horizontal {
            background: #f3f6fa;
            height: 10px;
            margin: 2px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: #c7d2df;
            border-radius: 5px;
            min-width: 28px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0;
            border: none;
            background: transparent;
        }
        QListWidget::item {
            padding: 6px 4px;
        }
        QListWidget::item:selected {
            background: #eff7f3;
            color: #162033;
        }
        QTabWidget::pane {
            border: 1px solid #dbe3ee;
            border-radius: 8px;
            background: #ffffff;
            top: -1px;
        }
        QTabBar::tab {
            background: transparent;
            border: none;
            padding: 10px 14px;
            color: #6b7484;
            margin-right: 6px;
        }
        QTabBar::tab:selected {
            color: #1f9d68;
            border-bottom: 2px solid #1f9d68;
            font-weight: 600;
        }
        QHeaderView::section {
            background: #f4f7fb;
            border: none;
            border-bottom: 1px solid #dbe3ee;
            padding: 8px;
            font-weight: 600;
        }
        QStatusBar {
            background: #ffffff;
            border-top: 1px solid #dbe3ee;
        }
        """
    )
