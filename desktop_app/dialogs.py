from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QPushButton,
)

from .widgets import DropdownSelect


class CorpusTargetDialog(QDialog):
    def __init__(self, corpus_names: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择导入目标")
        self.setModal(True)
        self.resize(420, 220)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel("把新文本导入到哪个知识库？")
        title.setProperty("sectionTitle", True)
        root.addWidget(title)

        self.mode_group = QButtonGroup(self)
        self.new_radio = QRadioButton("新建知识库")
        self.existing_radio = QRadioButton("导入到现有知识库")
        self.mode_group.addButton(self.new_radio)
        self.mode_group.addButton(self.existing_radio)
        self.new_radio.setChecked(True)

        radio_row = QHBoxLayout()
        radio_row.addWidget(self.new_radio)
        radio_row.addWidget(self.existing_radio)
        radio_row.addStretch(1)
        root.addLayout(radio_row)

        self.new_name_input = QLineEdit()
        self.new_name_input.setPlaceholderText("例如：哈利波特 / 庆余年 / 某系列第六卷")

        self.existing_combo = DropdownSelect(max_popup_height=180)
        self.existing_combo.addItems(corpus_names)
        self.existing_combo.setEnabled(bool(corpus_names))

        new_box = QWidget()
        new_layout = QVBoxLayout(new_box)
        new_layout.setContentsMargins(0, 0, 0, 0)
        new_layout.setSpacing(6)
        new_layout.addWidget(QLabel("新知识库名称"))
        new_layout.addWidget(self.new_name_input)
        root.addWidget(new_box)

        existing_box = QWidget()
        existing_layout = QVBoxLayout(existing_box)
        existing_layout.setContentsMargins(0, 0, 0, 0)
        existing_layout.setSpacing(6)
        existing_layout.addWidget(QLabel("现有知识库"))
        existing_layout.addWidget(self.existing_combo)
        root.addWidget(existing_box)
        self.existing_box = existing_box

        tip = QLabel("导入后只会重建目标知识库，不会动其他小说。")
        tip.setProperty("muted", True)
        tip.setWordWrap(True)
        root.addWidget(tip)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.new_radio.toggled.connect(self._sync_mode)
        self._sync_mode()

    def selected_corpus_name(self) -> str:
        if self.new_radio.isChecked():
            return self.new_name_input.text().strip()
        return self.existing_combo.currentText().strip()

    def accept(self) -> None:
        if self.selected_corpus_name():
            super().accept()

    def _sync_mode(self) -> None:
        is_new = self.new_radio.isChecked()
        self.new_name_input.setEnabled(is_new)
        self.existing_box.setEnabled(not is_new)


class SettingsDialog(QDialog):
    FIELD_LABELS = {
        "DEEPSEEK_API_KEY": "DeepSeek API Key",
        "DEEPSEEK_API_BASE": "DeepSeek API Base",
        "DEEPSEEK_MODEL": "默认问答模型",
        "SILICONFLOW_API_KEY": "Embedding API Key",
        "SILICONFLOW_API_BASE": "Embedding API Base",
        "RAG_EMBEDDING_MODEL": "Embedding 模型",
    }

    FIELD_HINTS = {
        "DEEPSEEK_API_KEY": "用于问题分析、重排、回答生成",
        "DEEPSEEK_API_BASE": "例如：https://api.deepseek.com",
        "DEEPSEEK_MODEL": "例如：dsv4pro / dsv4flash / deepseek-v4-pro",
        "SILICONFLOW_API_KEY": "用于向量化建库和检索 query embedding",
        "SILICONFLOW_API_BASE": "例如：https://api.siliconflow.cn/v1",
        "RAG_EMBEDDING_MODEL": "例如：BAAI/bge-m3",
    }

    def __init__(self, settings: dict[str, str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("运行设置")
        self.setModal(True)
        self.resize(620, 430)
        self.inputs: dict[str, QLineEdit] = {}
        self.show_debug_checkbox = QCheckBox("显示调试面板")
        self.show_debug_checkbox.setChecked(str(settings.get("STORYRAG_SHOW_DEBUG_PANEL", "0")).strip() == "1")

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        title = QLabel("API 与模型设置")
        title.setProperty("sectionTitle", True)
        root.addWidget(title)

        tip = QLabel("保存后会写入项目 .env，并立即应用到本次运行。建库模型和提问模型仍可在主界面单独切换。")
        tip.setProperty("muted", True)
        tip.setWordWrap(True)
        root.addWidget(tip)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        for key, label in self.FIELD_LABELS.items():
            field = QLineEdit()
            field.setText(settings.get(key, ""))
            field.setPlaceholderText(self.FIELD_HINTS.get(key, ""))
            if key.endswith("API_KEY"):
                field.setEchoMode(QLineEdit.Password)
            self.inputs[key] = field
            form.addRow(label, field)
        form.addRow("高级", self.show_debug_checkbox)

        root.addLayout(form, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def settings(self) -> dict[str, str]:
        values = {key: field.text().strip() for key, field in self.inputs.items()}
        values["STORYRAG_SHOW_DEBUG_PANEL"] = "1" if self.show_debug_checkbox.isChecked() else "0"
        return values


class AliasManagerDialog(QDialog):
    def __init__(self, corpus_names: list[str], aliases_by_corpus: dict[str, list[dict]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("别名管理")
        self.setModal(True)
        self.resize(680, 460)
        self.aliases_by_corpus = {
            name: [dict(item) for item in aliases_by_corpus.get(name, [])]
            for name in corpus_names
        }
        self._current_corpus = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel("按知识库维护检索别名")
        title.setProperty("sectionTitle", True)
        root.addWidget(title)

        tip = QLabel("例如：老八 -> 八奈见，老马 -> 天爱星。保存后会立刻用于查询扩展和关键词召回。")
        tip.setProperty("muted", True)
        tip.setWordWrap(True)
        root.addWidget(tip)

        top = QHBoxLayout()
        top.addWidget(QLabel("知识库"))
        self.corpus_combo = DropdownSelect(max_popup_height=180)
        self.corpus_combo.addItems(corpus_names)
        self.corpus_combo.currentTextChanged.connect(self._load_current_corpus)
        top.addWidget(self.corpus_combo, 1)
        root.addLayout(top)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["别名", "标准名", "备注"])
        root.addWidget(self.table, 1)

        row_buttons = QHBoxLayout()
        add_button = QPushButton("新增")
        add_button.clicked.connect(self._add_row)
        remove_button = QPushButton("删除选中")
        remove_button.clicked.connect(self._remove_selected_rows)
        row_buttons.addWidget(add_button)
        row_buttons.addWidget(remove_button)
        row_buttons.addStretch(1)
        root.addLayout(row_buttons)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load_current_corpus()

    def aliases(self) -> dict[str, list[dict]]:
        self._save_current_corpus()
        return self.aliases_by_corpus

    def _load_current_corpus(self) -> None:
        if self._current_corpus:
            self._save_corpus_from_table(self._current_corpus)
        corpus_name = self.corpus_combo.currentText().strip()
        self._current_corpus = corpus_name
        self.table.setRowCount(0)
        for item in self.aliases_by_corpus.get(corpus_name, []):
            self._add_row(item)

    def _save_current_corpus(self) -> None:
        corpus_name = self._current_corpus or self.corpus_combo.currentText().strip()
        self._save_corpus_from_table(corpus_name)

    def _save_corpus_from_table(self, corpus_name: str) -> None:
        if not corpus_name:
            return
        rows = []
        for row in range(self.table.rowCount()):
            alias = self._table_text(row, 0)
            canonical = self._table_text(row, 1)
            note = self._table_text(row, 2)
            if alias and canonical:
                rows.append({"alias": alias, "canonical": canonical, "note": note})
        self.aliases_by_corpus[corpus_name] = rows

    def _add_row(self, item: dict | None = None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        payload = item or {}
        for col, key in enumerate(("alias", "canonical", "note")):
            self.table.setItem(row, col, QTableWidgetItem(str(payload.get(key, ""))))

    def _remove_selected_rows(self) -> None:
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def _table_text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item is not None else ""

    def accept(self) -> None:
        self._save_current_corpus()
        super().accept()
