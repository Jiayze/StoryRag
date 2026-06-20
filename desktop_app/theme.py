from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f4f6f8"))
    palette.setColor(QPalette.WindowText, QColor("#1d2733"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f0f3f7"))
    palette.setColor(QPalette.Text, QColor("#1d2733"))
    palette.setColor(QPalette.Button, QColor("#ffffff"))
    palette.setColor(QPalette.ButtonText, QColor("#1d2733"))
    palette.setColor(QPalette.Highlight, QColor("#2f7d7e"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.PlaceholderText, QColor("#7a8492"))
    app.setPalette(palette)

    app.setStyleSheet(
        """
        QWidget {
            color: #1d2733;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 13px;
        }
        QMainWindow, QDialog {
            background: #f4f6f8;
        }
        QFrame[card="true"] {
            background: #ffffff;
            border: 1px solid #dde3eb;
            border-radius: 8px;
        }
        QFrame[headerBar="true"] {
            background: #ffffff;
            border: 1px solid #dde3eb;
            border-radius: 8px;
        }
        QFrame[statBlock="true"] {
            background: #f8fafc;
            border: 1px solid #e5eaf1;
            border-radius: 7px;
        }
        QFrame[chatBubbleUser="true"] {
            background: #e2f2f1;
            border: 1px solid #aad2cf;
            border-radius: 12px;
        }
        QFrame[chatBubbleAssistant="true"] {
            background: #ffffff;
            border: 1px solid #dce3ec;
            border-radius: 12px;
        }
        QFrame[chatBubbleAssistant="true"][blocked="true"] {
            background: #fff7ed;
            border: 1px solid #efc9a7;
        }
        QFrame[composer="true"] {
            background: #ffffff;
            border: 1px solid #d7e0ea;
            border-radius: 8px;
        }
        QLabel[chatRole="true"] {
            color: #697586;
            font-size: 12px;
            font-weight: 600;
        }
        QFrame[evidenceCard="true"] {
            background: #fbfcfe;
            border: 1px solid #dde3eb;
            border-radius: 8px;
        }
        QLabel[evidenceTitle="true"] {
            color: #243142;
            font-weight: 600;
        }
        QLabel[evidenceDetail="true"] {
            background: #f5f7fa;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px;
            line-height: 1.45;
        }
        QLabel[muted="true"] {
            color: #697586;
        }
        QLabel[caption="true"] {
            color: #7a8492;
            font-size: 12px;
        }
        QLabel[sectionTitle="true"] {
            color: #263445;
            font-size: 13px;
            font-weight: 700;
        }
        QLabel[heroTitle="true"] {
            color: #16202f;
            font-size: 22px;
            font-weight: 700;
        }
        QLabel[badge="true"] {
            background: #eef8f5;
            border: 1px solid #c9e4de;
            border-radius: 13px;
            padding: 5px 10px;
            color: #216f68;
            font-weight: 700;
        }
        QLabel[statLabel="true"] {
            color: #6f7a89;
            font-size: 12px;
        }
        QLabel[statValue="true"] {
            color: #172232;
            font-size: 20px;
            font-weight: 800;
        }
        QLabel[warningText="true"] {
            color: #8a5a13;
            background: #fff7e8;
            border: 1px solid #f1d8a7;
            border-radius: 6px;
            padding: 6px 8px;
        }
        QPushButton {
            background: #ffffff;
            border: 1px solid #cfd7e2;
            border-radius: 8px;
            padding: 8px 12px;
            min-height: 20px;
        }
        QPushButton:hover {
            border-color: #2f7d7e;
            background: #f8fbfb;
        }
        QPushButton:pressed {
            background: #edf5f5;
        }
        QPushButton:disabled {
            color: #9aa4b2;
            border-color: #dce3ec;
            background: #f7f9fb;
        }
        QPushButton[primary="true"] {
            background: #2f7d7e;
            color: #ffffff;
            border: 1px solid #2f7d7e;
            font-weight: 600;
        }
        QPushButton[primary="true"]:hover {
            background: #256b6c;
            border-color: #256b6c;
        }
        QPushButton[secondary="true"] {
            background: #eef8f5;
            border: 1px solid #c9e4de;
            color: #216f68;
            font-weight: 600;
        }
        QPushButton[quiet="true"] {
            background: transparent;
            border: 1px solid #d8e0ea;
        }
        QPushButton[danger="true"] {
            color: #9a4b25;
            background: #fff5ef;
            border: 1px solid #f0cbb8;
        }
        QPushButton[compact="true"] {
            padding: 5px 9px;
            min-height: 18px;
            border-radius: 6px;
        }
        QFrame#dropdownHeader {
            background: #ffffff;
            border: 1px solid #d8e0ea;
            border-radius: 8px;
            min-height: 34px;
        }
        QFrame#dropdownHeader:hover {
            border-color: #2f7d7e;
        }
        QFrame#dropdownHeader[open="true"] {
            border-color: #2f7d7e;
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }
        QFrame#dropdownHeader[open="true"][opensUp="true"] {
            border-color: #2f7d7e;
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        QLabel#dropdownLabel {
            background: transparent;
            color: #1d2733;
            padding-left: 0;
        }
        QLabel#dropdownArrow {
            background: transparent;
            color: #586576;
            font-size: 13px;
            font-weight: 700;
        }
        QFrame#dropdownPopup {
            background: #ffffff;
            border: 1px solid #2f7d7e;
            border-top: none;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        QFrame#dropdownPopup[opensUp="true"] {
            border: 1px solid #2f7d7e;
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
            background: #eef8f5;
            border: none;
        }
        QPushButton#dropdownOption[selected="true"] {
            background: #e2f2f1;
            color: #216f68;
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
            border: 1px solid #d8e0ea;
            border-radius: 8px;
            padding: 6px;
        }
        QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QTextBrowser:focus, QComboBox:focus, QListWidget:focus, QTableWidget:focus, QTreeWidget:focus {
            border: 1px solid #2f7d7e;
        }
        QPlainTextEdit, QTextEdit, QTextBrowser {
            selection-background-color: #cfe7e4;
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
            border: 1px solid #2f7d7e;
            border-radius: 8px;
            padding: 4px;
            outline: 0;
            selection-background-color: #eef8f5;
            selection-color: #1d2733;
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
            color: #1d2733;
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
            background: #f0f3f7;
            width: 10px;
            margin: 2px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #c6d0dc;
            border-radius: 5px;
            min-height: 28px;
        }
        QScrollBar::handle:vertical:hover {
            background: #9eabba;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
            border: none;
            background: transparent;
        }
        QScrollBar:horizontal {
            background: #f0f3f7;
            height: 10px;
            margin: 2px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: #c6d0dc;
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
            background: #eef8f5;
            color: #1d2733;
        }
        QTabWidget::pane {
            border: 1px solid #dde3eb;
            border-radius: 8px;
            background: #ffffff;
            top: -1px;
        }
        QTabBar::tab {
            background: transparent;
            border: none;
            padding: 10px 14px;
            color: #697586;
            margin-right: 6px;
        }
        QTabBar::tab:selected {
            color: #216f68;
            border-bottom: 2px solid #2f7d7e;
            font-weight: 600;
        }
        QHeaderView::section {
            background: #f4f6f8;
            border: none;
            border-bottom: 1px solid #dde3eb;
            padding: 8px;
            font-weight: 600;
        }
        QSplitter::handle {
            background: transparent;
        }
        QSplitter::handle:hover {
            background: #e5eaf1;
        }
        QStatusBar {
            background: #ffffff;
            border-top: 1px solid #dde3eb;
        }
        """
    )
