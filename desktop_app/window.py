from __future__ import annotations

import html
import traceback
from functools import partial
from typing import Any

from PySide6.QtCore import QEvent, Qt, QThreadPool
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app_services import (
    ask_story_question,
    available_corpus_names,
    export_knowledge_package,
    import_knowledge_package,
    inspect_knowledge_package,
    load_alias_entries,
    load_runtime_settings,
    load_workspace_snapshot,
    preview_incremental_update,
    rebuild_knowledge_base_from_local_files,
    rebuild_selected_corpus,
    save_runtime_settings,
    save_alias_entries,
    update_corpus_incrementally,
)
from qa.context import format_selected_contexts
from retrieval import format_debug_table
from core import get_logger

from .dialogs import AliasManagerDialog, CorpusTargetDialog, SettingsDialog
from .widgets import DropdownSelect
from .workers import FunctionWorker

logger = get_logger(__name__)


class StoryRagWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StoryRAG Desktop")
        self.resize(1660, 980)

        self.thread_pool = QThreadPool.globalInstance()
        self.chat_history: list[dict[str, str]] = []
        self.current_followup_options: list[dict] = []
        self.pinned_contexts: list[dict] = []
        self.active_workers: list[FunctionWorker] = []
        self.chat_message_index = 0
        self.query_records: list[dict[str, Any]] = []
        self.pending_question: str = ""
        self.pending_selected_contexts: list[dict] = []
        self.pending_search_scope: dict[str, Any] = {}
        self.evidence_cards: list[dict[str, Any]] = []
        self.pinned_cards: list[dict[str, Any]] = []
        self.scope_catalog: dict[str, dict[str, Any]] = {}
        self.has_corpora = False
        self.busy = False

        self._build_ui()
        self.refresh_workspace()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([320, 820, 460])
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)

        status_bar = QStatusBar()
        self.status_label = QLabel("就绪")
        status_bar.addWidget(self.status_label)
        self.setStatusBar(status_bar)

        refresh_action = QAction("刷新", self)
        refresh_action.triggered.connect(self.refresh_workspace)
        self.addAction(refresh_action)

    def _build_header(self) -> QWidget:
        frame = self._card()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        title_box = QVBoxLayout()
        title = QLabel("StoryRAG Desktop")
        title.setProperty("heroTitle", True)
        subtitle = QLabel("本地小说知识库、多语料检索、证据约束问答")
        subtitle.setProperty("muted", True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box, 1)

        self.runtime_badge = QLabel("Workspace Ready")
        self.runtime_badge.setStyleSheet(
            "background:#eff7f3;border:1px solid #cfe8dc;border-radius:14px;padding:6px 10px;color:#1f9d68;font-weight:600;"
        )
        layout.addWidget(self.runtime_badge, 0, Qt.AlignRight | Qt.AlignVCenter)

        self.settings_button = QPushButton("设置")
        self.settings_button.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_button)
        return frame

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        stats_card = self._card()
        stats_layout = QGridLayout(stats_card)
        stats_layout.setContentsMargins(16, 16, 16, 16)
        stats_layout.setHorizontalSpacing(12)
        stats_layout.setVerticalSpacing(10)
        stats_layout.addWidget(self._section_label("工作区概览"), 0, 0, 1, 2)
        self.indexed_files_value = QLabel("--")
        self.vector_chunks_value = QLabel("--")
        self.processed_chunks_value = QLabel("--")
        self.relations_value = QLabel("--")
        for label, value, row, col in (
            ("文本文件", self.indexed_files_value, 1, 0),
            ("向量块", self.vector_chunks_value, 1, 1),
            ("处理块", self.processed_chunks_value, 2, 0),
            ("关系数", self.relations_value, 2, 1),
        ):
            block = QVBoxLayout()
            top = QLabel(label)
            top.setProperty("muted", True)
            value.setStyleSheet("font-size:20px;font-weight:700;")
            block.addWidget(top)
            block.addWidget(value)
            stats_layout.addLayout(block, row, col)
        layout.addWidget(stats_card)

        corpus_card = self._card()
        corpus_layout = QVBoxLayout(corpus_card)
        corpus_layout.setContentsMargins(16, 16, 16, 16)
        corpus_layout.setSpacing(10)
        corpus_layout.addWidget(self._section_label("搜索范围"))
        corpus_tip = QLabel("可多选书和卷。勾选具体卷时会严格排除未标卷旧内容。")
        corpus_tip.setProperty("muted", True)
        corpus_tip.setWordWrap(True)
        corpus_layout.addWidget(corpus_tip)
        self.scope_warning_label = QLabel("")
        self.scope_warning_label.setProperty("muted", True)
        self.scope_warning_label.setWordWrap(True)
        corpus_layout.addWidget(self.scope_warning_label)
        self.corpus_tree = QTreeWidget()
        self.corpus_tree.setHeaderHidden(True)
        self.corpus_tree.setSelectionMode(QAbstractItemView.NoSelection)
        self.corpus_tree.setFocusPolicy(Qt.NoFocus)
        self.corpus_tree.setMinimumHeight(190)
        self.corpus_tree.setMaximumHeight(340)
        self.corpus_tree.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.corpus_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.corpus_tree.itemChanged.connect(self._handle_scope_item_changed)
        corpus_layout.addWidget(self.corpus_tree)
        corpus_buttons = QHBoxLayout()
        select_all_button = QPushButton("全选")
        select_all_button.clicked.connect(lambda: self._set_all_corpora_checked(True))
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(lambda: self._set_all_corpora_checked(False))
        corpus_buttons.addWidget(select_all_button)
        corpus_buttons.addWidget(clear_button)
        corpus_layout.addLayout(corpus_buttons)
        layout.addWidget(corpus_card)

        ingest_card = self._card()
        ingest_layout = QVBoxLayout(ingest_card)
        ingest_layout.setContentsMargins(16, 16, 16, 16)
        ingest_layout.setSpacing(10)
        ingest_layout.addWidget(self._section_label("导入与重建"))
        self.doc_dir_label = QLabel("--")
        self.doc_dir_label.setProperty("muted", True)
        self.doc_dir_label.setWordWrap(True)
        ingest_layout.addWidget(self.doc_dir_label)
        build_model_row = QHBoxLayout()
        build_model_row.addWidget(QLabel("建库模型"))
        self.build_model_combo = DropdownSelect()
        self._populate_model_combo(self.build_model_combo)
        build_model_row.addWidget(self.build_model_combo, 1)
        ingest_layout.addLayout(build_model_row)
        self.import_button = QPushButton("导入本地 TXT 并建库")
        self.import_button.setProperty("primary", True)
        self.import_button.clicked.connect(self.import_files)
        ingest_layout.addWidget(self.import_button)
        self.incremental_button = QPushButton("增量更新现有知识库")
        self.incremental_button.clicked.connect(self.incremental_update)
        ingest_layout.addWidget(self.incremental_button)
        package_row = QHBoxLayout()
        self.export_package_button = QPushButton("导出建库包")
        self.export_package_button.clicked.connect(self.export_package)
        self.import_package_button = QPushButton("导入建库包")
        self.import_package_button.clicked.connect(self.import_package)
        package_row.addWidget(self.export_package_button)
        package_row.addWidget(self.import_package_button)
        ingest_layout.addLayout(package_row)
        rebuild_row = QHBoxLayout()
        self.rebuild_combo = DropdownSelect()
        rebuild_row.addWidget(self.rebuild_combo, 1)
        self.rebuild_button = QPushButton("强制重建")
        self.rebuild_button.clicked.connect(self.rebuild_corpus)
        rebuild_row.addWidget(self.rebuild_button)
        ingest_layout.addLayout(rebuild_row)
        self.alias_button = QPushButton("管理别名")
        self.alias_button.clicked.connect(self.manage_aliases)
        ingest_layout.addWidget(self.alias_button)
        layout.addWidget(ingest_card)

        session_card = self._card()
        session_layout = QVBoxLayout(session_card)
        session_layout.setContentsMargins(16, 16, 16, 16)
        session_layout.setSpacing(10)
        session_layout.addWidget(self._section_label("追问上下文"))
        self.pinned_summary_label = QLabel("已固定证据 0 条")
        self.pinned_summary_label.setProperty("muted", True)
        session_layout.addWidget(self.pinned_summary_label)
        session_layout.addWidget(QLabel("提问时会自动带上已固定证据，以及本轮右侧勾选证据。"))
        self.clear_pinned_button = QPushButton("清空固定证据")
        self.clear_pinned_button.clicked.connect(self.clear_pinned_contexts)
        session_layout.addWidget(self.clear_pinned_button)
        layout.addWidget(session_card)

        layout.addStretch(1)
        scroll.setWidget(wrapper)
        return scroll

    def _build_center_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_chat_tab(), "问答工作台")
        tabs.addTab(self._build_overview_tab(), "知识库总览")
        return tabs

    def _build_chat_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_view = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_view)
        self.chat_layout.setContentsMargins(14, 14, 14, 14)
        self.chat_layout.setSpacing(14)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_view)
        layout.addWidget(self.chat_scroll, 1)

        composer = self._card()
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(16, 16, 16, 16)
        composer_layout.setSpacing(10)

        hint = QLabel("提问时会结合本轮检索结果、固定证据，以及最近几轮问题历史。")
        hint.setProperty("muted", True)
        composer_layout.addWidget(hint)

        self.question_input = QPlainTextEdit()
        self.question_input.setPlaceholderText("需要先建库，点击这里选择 TXT 并建库")
        self.question_input.installEventFilter(self)
        self.question_input.setFixedHeight(110)
        composer_layout.addWidget(self.question_input)

        composer_buttons = QHBoxLayout()
        composer_buttons.addWidget(QLabel("提问模型"))
        self.ask_model_combo = DropdownSelect(popup_direction="up")
        self._populate_model_combo(self.ask_model_combo)
        composer_buttons.addWidget(self.ask_model_combo)
        self.send_button = QPushButton("发送问题")
        self.send_button.setProperty("primary", True)
        self.send_button.clicked.connect(self.ask_question)
        clear_chat_button = QPushButton("清空对话")
        clear_chat_button.clicked.connect(self.clear_chat)
        composer_buttons.addWidget(self.send_button)
        composer_buttons.addWidget(clear_chat_button)
        composer_buttons.addStretch(1)
        composer_layout.addLayout(composer_buttons)

        layout.addWidget(composer)
        return wrapper

    def _build_overview_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.corpora_table = QTableWidget(0, 4)
        self.corpora_table.setHorizontalHeaderLabels(["知识库", "文档数", "章节数", "块数"])
        self.corpora_table.verticalHeader().setVisible(False)
        self.corpora_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.corpora_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.corpora_table, 1)

        self.manifest_view = QTextBrowser()
        layout.addWidget(self.manifest_view, 1)
        return wrapper

    def _build_right_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_evidence_tab(), "证据")
        tabs.addTab(self._build_debug_tab(), "调试")
        tabs.addTab(self._build_details_tab(), "详情")
        tabs.addTab(self._build_log_tab(), "日志")
        return tabs

    def eventFilter(self, watched, event) -> bool:
        if watched is getattr(self, "question_input", None) and event.type() == QEvent.MouseButtonPress:
            if not self.has_corpora and not self.busy:
                self.import_files()
                return True
        return super().eventFilter(watched, event)

    def _build_evidence_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(self._section_label("查询历史"))
        self.query_record_list = QListWidget()
        self.query_record_list.setMaximumHeight(150)
        self.query_record_list.currentRowChanged.connect(self.restore_query_record)
        layout.addWidget(self.query_record_list)

        layout.addWidget(self._section_label("固定证据"))
        self.pinned_scroll = QScrollArea()
        self.pinned_scroll.setWidgetResizable(True)
        self.pinned_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.pinned_scroll.setMaximumHeight(210)
        self.pinned_view = QWidget()
        self.pinned_layout = QVBoxLayout(self.pinned_view)
        self.pinned_layout.setContentsMargins(0, 0, 0, 0)
        self.pinned_layout.setSpacing(8)
        self.pinned_layout.addStretch(1)
        self.pinned_scroll.setWidget(self.pinned_view)
        layout.addWidget(self.pinned_scroll, 1)
        pinned_buttons = QHBoxLayout()
        remove_pinned_button = QPushButton("移除选中")
        remove_pinned_button.clicked.connect(self.remove_selected_pinned_contexts)
        pinned_buttons.addWidget(remove_pinned_button)
        pinned_buttons.addWidget(self.clear_pinned_button_proxy())
        layout.addLayout(pinned_buttons)

        layout.addWidget(self._section_label("本轮检索证据"))
        self.followup_scroll = QScrollArea()
        self.followup_scroll.setWidgetResizable(True)
        self.followup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.followup_view = QWidget()
        self.followup_layout = QVBoxLayout(self.followup_view)
        self.followup_layout.setContentsMargins(0, 0, 0, 0)
        self.followup_layout.setSpacing(8)
        self.followup_layout.addStretch(1)
        self.followup_scroll.setWidget(self.followup_view)
        layout.addWidget(self.followup_scroll, 2)
        followup_buttons = QHBoxLayout()
        pin_button = QPushButton("固定勾选证据")
        pin_button.clicked.connect(self.pin_checked_followup_contexts)
        followup_buttons.addWidget(pin_button)
        followup_buttons.addStretch(1)
        layout.addLayout(followup_buttons)
        return wrapper

    def _build_debug_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.debug_table = QTableWidget(0, 0)
        self.debug_table.verticalHeader().setVisible(False)
        self.debug_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.debug_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.debug_table)
        return wrapper

    def _build_details_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(12, 12, 12, 12)
        self.details_browser = QTextBrowser()
        layout.addWidget(self.details_browser)
        return wrapper

    def _build_log_tab(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(12, 12, 12, 12)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        return wrapper

    def refresh_workspace(self) -> None:
        snapshot = load_workspace_snapshot()
        manifest = snapshot.get("manifest", {})
        corpora = snapshot.get("corpora", [])
        self.scope_catalog = snapshot.get("scope_catalog", {}) or {}
        self.has_corpora = bool(corpora)

        self.doc_dir_label.setText(f"文档目录：{snapshot.get('doc_dir', '--')}")
        self.indexed_files_value.setText(str(snapshot.get("indexed_file_count", 0)))
        self.vector_chunks_value.setText(str(snapshot.get("vector_chunk_count", 0)))
        self.processed_chunks_value.setText(str(manifest.get("chunk_count", 0)))
        self.relations_value.setText(str(manifest.get("relation_count", 0)))
        self.runtime_badge.setText(f"{len(corpora)} 个知识库")

        self._populate_corpus_selector(corpora)
        self._populate_rebuild_combo(corpora)
        self._populate_overview(corpora, manifest)
        self._update_question_placeholder()
        self._append_log("已刷新工作区状态。")

    def open_settings(self) -> None:
        dialog = SettingsDialog(load_runtime_settings(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            save_runtime_settings(dialog.settings())
        except Exception:
            traceback_text = traceback.format_exc()
            self._append_log(traceback_text)
            QMessageBox.critical(
                self,
                "保存设置失败",
                traceback_text.splitlines()[-1] if traceback_text else "未知错误",
            )
            return
        self._append_log("运行设置已保存并应用。")
        QMessageBox.information(self, "设置已保存", "API 与模型设置已写入 .env，并应用到当前运行。")

    def import_files(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择 TXT 文件",
            "",
            "Text Files (*.txt)",
        )
        if not file_paths:
            return

        dialog = CorpusTargetDialog(available_corpus_names(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        corpus_name = dialog.selected_corpus_name()
        if not corpus_name:
            QMessageBox.warning(self, "缺少知识库名称", "请先提供知识库名称。")
            return

        self._run_async(
            rebuild_knowledge_base_from_local_files,
            file_paths,
            corpus_name=corpus_name,
            model=self._selected_build_model(),
            busy_text=f"正在导入并构建知识库：{corpus_name}",
            success_handler=partial(self._handle_import_success, corpus_name),
            error_title="导入失败",
        )

    def rebuild_corpus(self) -> None:
        corpus_name = self.rebuild_combo.currentText().strip()
        if not corpus_name:
            QMessageBox.information(self, "没有可重建的知识库", "请先导入至少一个知识库。")
            return
        reply = QMessageBox.question(
            self,
            "确认强制重建",
            f"将重新处理“{corpus_name}”下全部 TXT，并重新生成向量。这个操作成本较高。\n是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._run_async(
            rebuild_selected_corpus,
            corpus_name,
            model=self._selected_build_model(),
            busy_text=f"正在重建知识库：{corpus_name}",
            success_handler=partial(self._handle_rebuild_success, corpus_name),
            error_title="重建失败",
        )

    def incremental_update(self) -> None:
        corpus_names = available_corpus_names()
        if not corpus_names:
            QMessageBox.information(self, "没有可更新的知识库", "请先创建至少一个知识库。")
            return
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择要增量导入的 TXT 文件",
            "",
            "Text Files (*.txt)",
        )
        if not file_paths:
            return
        dialog = CorpusTargetDialog(corpus_names, self)
        dialog.new_radio.setEnabled(False)
        dialog.existing_radio.setChecked(True)
        dialog.new_name_input.setEnabled(False)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        corpus_name = dialog.selected_corpus_name()
        if not corpus_name:
            return
        try:
            preview = preview_incremental_update(file_paths, corpus_name=corpus_name)
        except Exception:
            preview = {}
        message = (
            f"将增量更新“{corpus_name}”。\n\n"
            f"新增文件：{preview.get('added_files', 0)}\n"
            f"更新文件：{preview.get('updated_files', 0)}\n"
            f"跳过未变化文件：{preview.get('skipped_files', 0)}\n"
            f"预计新增/重建 chunks：{preview.get('estimated_chunks', 0)}\n"
            f"会调用 DS 预处理：{'是' if preview.get('will_call_deepseek') else '否'}\n"
            f"会调用 Embedding：{'是' if preview.get('will_call_embedding') else '否'}\n\n"
            "未变化文件不会重跑高成本步骤。是否继续？"
        )
        reply = QMessageBox.question(
            self,
            "确认增量更新",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return
        self._run_async(
            update_corpus_incrementally,
            file_paths,
            corpus_name=corpus_name,
            model=self._selected_build_model(),
            busy_text=f"正在增量更新知识库：{corpus_name}",
            success_handler=partial(self._handle_incremental_success, corpus_name),
            error_title="增量更新失败",
        )

    def manage_aliases(self) -> None:
        corpus_names = available_corpus_names()
        if not corpus_names:
            QMessageBox.information(self, "没有可管理的知识库", "请先创建至少一个知识库。")
            return
        aliases = load_alias_entries()
        dialog = AliasManagerDialog(corpus_names, aliases if isinstance(aliases, dict) else {}, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for corpus_name, entries in dialog.aliases().items():
            save_alias_entries(corpus_name, entries)
        self._append_log("别名表已保存，后续检索会自动使用。")
        QMessageBox.information(self, "别名已保存", "别名表已保存，后续检索会自动扩展关键词。")

    def export_package(self) -> None:
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 StoryRAG 建库包",
            "storyrag_kb_package.zip",
            "StoryRAG Package (*.zip)",
        )
        if not target_path:
            return
        self._run_async(
            export_knowledge_package,
            target_path,
            corpus_names=self._selected_corpus_names(),
            busy_text="正在导出建库包",
            success_handler=self._handle_export_package_success,
            error_title="导出失败",
        )

    def import_package(self) -> None:
        package_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入 StoryRAG 建库包",
            "",
            "StoryRAG Package (*.zip)",
        )
        if not package_path:
            return
        try:
            info = inspect_knowledge_package(package_path)
        except Exception:
            traceback_text = traceback.format_exc()
            self._append_log(traceback_text)
            QMessageBox.critical(self, "导入失败", "只能导入通过本软件导出的 StoryRAG 建库包。")
            return
        conflicts = list(info.get("conflicts", []))
        corpus_names = ", ".join(info.get("corpus_names", []))
        message = (
            "只能导入通过本软件导出的建库包。\n\n"
            f"包内知识库：{corpus_names or '未知'}\n"
            "导入会恢复原始 TXT、预处理结果和向量索引。"
        )
        if conflicts:
            message += f"\n\n检测到同名知识库：{', '.join(conflicts)}\n继续导入将覆盖这些本地知识库。"
        reply = QMessageBox.question(
            self,
            "确认导入建库包",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._run_async(
            import_knowledge_package,
            package_path,
            overwrite_corpora=bool(conflicts),
            busy_text="正在导入建库包",
            success_handler=self._handle_import_package_success,
            error_title="导入失败",
        )

    def ask_question(self) -> None:
        question = self.question_input.toPlainText().strip()
        if not question:
            return

        selected_contexts = self._selected_context_payloads()
        search_scope = self._selected_search_scope()
        if not search_scope.get("corpora"):
            QMessageBox.information(self, "未选择搜索范围", "请先在左侧至少勾选一个知识库或卷册。")
            return
        corpus_names = list(search_scope.get("corpora", []))
        self.pending_question = question
        self.pending_selected_contexts = list(selected_contexts)
        self.pending_search_scope = dict(search_scope)
        self._append_chat("user", question)
        self.chat_history.append({"role": "user", "content": question})
        self.question_input.clear()

        self._run_async(
            ask_story_question,
            question=question,
            corpus_names=corpus_names,
            search_scope=search_scope,
            question_history=self.chat_history,
            selected_contexts=selected_contexts,
            model=self._selected_ask_model(),
            busy_text="正在检索证据并生成回答",
            success_handler=self._handle_answer_success,
            error_title="问答失败",
        )

    def clear_chat(self) -> None:
        self.chat_history.clear()
        self.chat_message_index = 0
        self._clear_chat_widgets()
        self.details_browser.clear()
        self.debug_table.clearContents()
        self.debug_table.setRowCount(0)
        self._clear_followup_cards()
        self.current_followup_options.clear()
        self._clear_query_records()
        self._append_log("已清空当前对话。")

    def pin_checked_followup_contexts(self) -> None:
        added = 0
        known_ids = {str(item.get("option_id")) for item in self.pinned_contexts}
        for card in self.evidence_cards:
            checkbox = card.get("checkbox")
            if checkbox is None or not checkbox.isChecked():
                continue
            payload = card.get("payload") or {}
            option_id = str(payload.get("option_id", ""))
            if not option_id or option_id in known_ids:
                continue
            self.pinned_contexts.append(payload)
            known_ids.add(option_id)
            added += 1
        self._refresh_pinned_list()
        if added:
            self._append_log(f"已固定 {added} 条证据，供后续追问复用。")

    def clear_pinned_contexts(self) -> None:
        self.pinned_contexts.clear()
        self._refresh_pinned_list()
        self._append_log("已清空固定证据。")

    def remove_selected_pinned_contexts(self) -> None:
        selected_ids = {
            str((card.get("payload") or {}).get("option_id", ""))
            for card in getattr(self, "pinned_cards", [])
            if card.get("checkbox") is not None and card["checkbox"].isChecked()
        }
        if not selected_ids:
            return
        self.pinned_contexts = [
            item for item in self.pinned_contexts if str(item.get("option_id", "")) not in selected_ids
        ]
        self._refresh_pinned_list()
        self._append_log(f"已移除 {len(selected_ids)} 条固定证据。")

    def _handle_import_success(self, corpus_name: str, result) -> None:
        saved_paths, file_count = result
        self.refresh_workspace()
        self._append_log(
            f"知识库“{corpus_name}”导入完成，新增 {len(saved_paths)} 个文件，当前总文本数 {file_count}。"
        )
        QMessageBox.information(
            self,
            "导入完成",
            f"知识库“{corpus_name}”已完成构建。\n导入文件数：{len(saved_paths)}\n当前总文本数：{file_count}",
        )

    def _handle_rebuild_success(self, corpus_name: str, file_count: int) -> None:
        self.refresh_workspace()
        self._append_log(f"知识库“{corpus_name}”重建完成，当前总文本数 {file_count}。")
        QMessageBox.information(
            self,
            "重建完成",
            f"知识库“{corpus_name}”已完成重建。\n当前总文本数：{file_count}",
        )

    def _handle_incremental_success(self, corpus_name: str, result: dict) -> None:
        self.refresh_workspace()
        message = (
            f"知识库“{corpus_name}”增量更新完成。\n"
            f"新增文件：{result.get('added_files', 0)}\n"
            f"更新文件：{result.get('updated_files', 0)}\n"
            f"跳过未变化文件：{result.get('skipped_files', 0)}\n"
            f"新增正文 chunks：{result.get('new_chunks', 0)}\n"
            f"写入向量 chunks：{result.get('written_chunks', 0)}\n"
            f"角色表重算：{'是' if result.get('role_index_rebuilt') else '否'}"
        )
        self._append_log(message)
        QMessageBox.information(self, "增量更新完成", message)

    def _handle_export_package_success(self, result: dict) -> None:
        self._append_log(
            f"建库包导出完成：{result.get('path')}，知识库 {result.get('corpus_count')} 个，chunks {result.get('chunk_count')}。"
        )
        QMessageBox.information(
            self,
            "导出完成",
            f"建库包已导出。\n路径：{result.get('path')}\n知识库：{result.get('corpus_count')} 个\nChunks：{result.get('chunk_count')}",
        )

    def _handle_import_package_success(self, result: dict) -> None:
        self.refresh_workspace()
        names = ", ".join(result.get("corpus_names", []))
        self._append_log(f"建库包导入完成：{names}")
        QMessageBox.information(self, "导入完成", f"已导入知识库：{names or '未知'}")

    def _handle_answer_success(self, payload: dict) -> None:
        print("[INFO] Desktop answer UI update started.")
        retrieval_result = payload["retrieval_result"]
        validated = payload["validated_response"]
        self.current_followup_options = payload.get("followup_context_options", [])
        print(f"[INFO] Desktop rendering {len(self.current_followup_options)} follow-up options.")
        self._store_query_record(
            question=self.pending_question,
            retrieval_result=retrieval_result,
            validated=validated,
            followup_options=self.current_followup_options,
            selected_contexts=self.pending_selected_contexts,
        )
        self._populate_followup_options(self.current_followup_options)

        answer_text = validated.answer
        self.chat_history.append({"role": "assistant", "content": answer_text})
        self._append_chat("assistant", answer_text, blocked=validated.is_blocked)
        self._render_answer_details(
            retrieval_result,
            validated,
            selected_contexts=self.pending_selected_contexts,
        )
        self._render_debug_table(format_debug_table(retrieval_result.chunks))
        self._append_log("问答完成，已刷新证据与调试面板。")

    def _run_async(self, fn, *args, busy_text: str, success_handler, error_title: str, **kwargs) -> None:
        if self.busy:
            return
        self.busy = True
        self._set_busy_state(True, busy_text)
        worker = FunctionWorker(fn, *args, **kwargs)
        self.active_workers.append(worker)
        worker.signals.result.connect(partial(self._handle_worker_success, success_handler))
        worker.signals.error.connect(partial(self._handle_worker_error, error_title))
        worker.signals.finished.connect(partial(self._handle_worker_finished, worker))
        self.thread_pool.start(worker)

    def _handle_worker_success(self, success_handler, result) -> None:
        print("[INFO] Desktop worker success signal received.")
        self._set_busy_state(False, "就绪")
        try:
            success_handler(result)
        except Exception:
            traceback_text = traceback.format_exc()
            self._append_log(traceback_text)
            QMessageBox.critical(
                self,
                "界面更新失败",
                traceback_text.splitlines()[-1] if traceback_text else "未知错误",
            )

    def _handle_worker_error(self, title: str, traceback_text: str) -> None:
        print("[ERROR] Desktop worker error signal received.")
        self._set_busy_state(False, "就绪")
        self._append_log(traceback_text)
        QMessageBox.critical(self, title, traceback_text.splitlines()[-1] if traceback_text else "未知错误")

    def _handle_worker_finished(self, worker: FunctionWorker) -> None:
        if worker in self.active_workers:
            self.active_workers.remove(worker)
        print(f"[INFO] Desktop worker finished. active_workers={len(self.active_workers)}")

    def _set_busy_state(self, busy: bool, text: str) -> None:
        self.busy = busy
        self.status_label.setText(text)
        self.send_button.setEnabled(not busy)
        self.import_button.setEnabled(not busy)
        self.incremental_button.setEnabled(not busy)
        self.export_package_button.setEnabled(not busy)
        self.import_package_button.setEnabled(not busy)
        self.rebuild_button.setEnabled(not busy)
        self.alias_button.setEnabled(not busy)
        self.settings_button.setEnabled(not busy)

    def _populate_corpus_selector(self, corpora: list[dict]) -> None:
        existing_scope = self._selected_search_scope() if hasattr(self, "corpus_tree") else {}
        existing_volumes = existing_scope.get("volumes", {}) if isinstance(existing_scope, dict) else {}
        existing_corpora = set(existing_scope.get("corpora", []) if isinstance(existing_scope, dict) else [])
        self.corpus_tree.blockSignals(True)
        self.corpus_tree.clear()
        names = [str(item.get("corpus_name", "")).strip() for item in corpora if str(item.get("corpus_name", "")).strip()]
        for name in names:
            root = QTreeWidgetItem([name])
            root.setFlags(root.flags() | Qt.ItemIsUserCheckable)
            root.setData(0, Qt.UserRole, {"type": "corpus", "corpus_name": name})
            root.setCheckState(0, Qt.Checked if not existing_corpora or name in existing_corpora else Qt.Unchecked)
            volumes = list((self.scope_catalog.get(name) or {}).get("volumes", []) or [])
            selected_volume_indices = {int(value) for value in existing_volumes.get(name, []) if str(value).isdigit()}
            has_real_volume = False
            for volume in volumes:
                volume_index = volume.get("volume_index")
                if volume_index is None:
                    continue
                has_real_volume = True
                label = f"{volume.get('volume_label') or f'第{volume_index}卷'} ({volume.get('chunk_count', 0)})"
                child = QTreeWidgetItem([label])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setData(0, Qt.UserRole, {"type": "volume", "corpus_name": name, "volume_index": int(volume_index)})
                if selected_volume_indices:
                    child.setCheckState(0, Qt.Checked if int(volume_index) in selected_volume_indices else Qt.Unchecked)
                else:
                    child.setCheckState(0, Qt.Checked)
                root.addChild(child)
            if not has_real_volume:
                child = QTreeWidgetItem(["全部内容（未标卷）"])
                child.setFlags(child.flags() & ~Qt.ItemIsUserCheckable)
                child.setData(0, Qt.UserRole, {"type": "unlabeled", "corpus_name": name})
                root.addChild(child)
            self.corpus_tree.addTopLevelItem(root)
            root.setExpanded(True)
        self.corpus_tree.blockSignals(False)
        self._update_scope_warning()
        self._update_question_placeholder()

    def _populate_rebuild_combo(self, corpora: list[dict]) -> None:
        current = self.rebuild_combo.currentText()
        self.rebuild_combo.clear()
        names = [str(item.get("corpus_name", "")).strip() for item in corpora if str(item.get("corpus_name", "")).strip()]
        self.rebuild_combo.addItems(names)
        if current and current in names:
            self.rebuild_combo.setCurrentText(current)

    def _populate_overview(self, corpora: list[dict], manifest: dict) -> None:
        rows = list(corpora)
        self.corpora_table.setRowCount(len(rows))
        for row, corpus in enumerate(rows):
            values = [
                str(corpus.get("corpus_name", "")),
                str(corpus.get("document_count", 0)),
                str(corpus.get("chapter_count", 0)),
                str(corpus.get("chunk_count", 0)),
            ]
            for col, value in enumerate(values):
                self.corpora_table.setItem(row, col, QTableWidgetItem(value))
        self.corpora_table.resizeColumnsToContents()

        summary_lines = [
            f"Pipeline Version: {manifest.get('pipeline_version', 'unknown')}",
            f"Generated At: {manifest.get('generated_at', 'unknown')}",
            f"Document Count: {manifest.get('document_count', 0)}",
            f"Chapter Count: {manifest.get('chapter_count', 0)}",
            f"Chunk Count: {manifest.get('chunk_count', 0)}",
            f"Relation Count: {manifest.get('relation_count', 0)}",
        ]
        self.manifest_view.setPlainText("\n".join(summary_lines))

    def _store_query_record(
        self,
        *,
        question: str,
        retrieval_result,
        validated,
        followup_options: list[dict],
        selected_contexts: list[dict],
    ) -> None:
        record = {
            "question": question or retrieval_result.query,
            "retrieval_result": retrieval_result,
            "validated": validated,
            "followup_options": list(followup_options),
            "selected_contexts": list(selected_contexts),
            "debug_rows": format_debug_table(retrieval_result.chunks),
        }
        self.query_records.append(record)
        self._refresh_query_record_list(select_last=True)

    def _refresh_query_record_list(self, *, select_last: bool = False) -> None:
        self.query_record_list.blockSignals(True)
        self.query_record_list.clear()
        for index, record in enumerate(self.query_records, start=1):
            question = _truncate_ui_text(str(record.get("question", "")).replace("\n", " "), 80)
            chunk_count = len(record.get("followup_options", []))
            self.query_record_list.addItem(f"{index}. {question}  ({chunk_count} chunks)")
        self.query_record_list.blockSignals(False)
        if select_last and self.query_records:
            self.query_record_list.setCurrentRow(len(self.query_records) - 1)

    def restore_query_record(self, row: int) -> None:
        if row < 0 or row >= len(self.query_records):
            return
        record = self.query_records[row]
        self.current_followup_options = list(record.get("followup_options", []))
        self._populate_followup_options(self.current_followup_options)
        self._render_answer_details(
            record["retrieval_result"],
            record["validated"],
            selected_contexts=record.get("selected_contexts", []),
        )
        self._render_debug_table(record.get("debug_rows", []))
        self._append_log(f"Restored query record #{row + 1}.")

    def _clear_query_records(self) -> None:
        self.query_records.clear()
        if hasattr(self, "query_record_list"):
            self.query_record_list.blockSignals(True)
            self.query_record_list.clear()
            self.query_record_list.blockSignals(False)

    def _populate_followup_options(self, options: list[dict]) -> None:
        self._clear_followup_cards()
        by_id = {
            str(option.get("option_id", "")): option
            for option in options
            if str(option.get("option_id", "")).strip()
        }
        for option in options:
            self._add_evidence_card(option, by_id=by_id)

    def _clear_followup_cards(self) -> None:
        self.evidence_cards.clear()
        if not hasattr(self, "followup_layout"):
            return
        while self.followup_layout.count() > 1:
            item = self.followup_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _clear_chat_widgets(self) -> None:
        if not hasattr(self, "chat_layout"):
            return
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_context_card(
        self,
        option: dict,
        *,
        by_id: dict[str, dict],
        target_layout: QVBoxLayout,
        registry: list[dict[str, Any]],
        checkbox_tooltip: str,
    ) -> None:
        card = QFrame()
        card.setProperty("evidenceCard", True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(8)

        header = QHBoxLayout()
        checkbox = QCheckBox()
        checkbox.setToolTip(checkbox_tooltip)
        checkbox.setMinimumSize(28, 28)
        checkbox.setFocusPolicy(Qt.NoFocus)
        header.addWidget(checkbox, 0, Qt.AlignTop)

        title = QLabel(_evidence_source_label(option))
        title.setWordWrap(True)
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(title, 1)

        toggle_button = QPushButton("展开")
        toggle_button.setFixedWidth(58)
        header.addWidget(toggle_button, 0, Qt.AlignTop)
        card_layout.addLayout(header)

        detail = QLabel(_evidence_detail_text(option, by_id=by_id))
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail.setVisible(False)
        detail.setStyleSheet(
            "background:#f8fafc;border:1px solid #e1e7ef;border-radius:6px;padding:8px;line-height:1.45;"
        )
        card_layout.addWidget(detail)

        def toggle_detail() -> None:
            visible = not detail.isVisible()
            detail.setVisible(visible)
            toggle_button.setText("收起" if visible else "展开")

        toggle_button.clicked.connect(toggle_detail)
        target_layout.insertWidget(max(0, target_layout.count() - 1), card)
        registry.append({"checkbox": checkbox, "payload": option, "card": card})

    def _add_evidence_card(self, option: dict, *, by_id: dict[str, dict]) -> None:
        self._add_context_card(
            option,
            by_id=by_id,
            target_layout=self.followup_layout,
            registry=self.evidence_cards,
            checkbox_tooltip="勾选后可固定为追问上下文",
        )
        return
        card = QFrame()
        card.setProperty("evidenceCard", True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(8)

        header = QHBoxLayout()
        checkbox = QCheckBox()
        checkbox.setToolTip("勾选后可固定为追问上下文")
        checkbox.setMinimumSize(28, 28)
        checkbox.setFocusPolicy(Qt.NoFocus)
        header.addWidget(checkbox, 0, Qt.AlignTop)

        title = QLabel(_evidence_source_label(option))
        title.setWordWrap(True)
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.addWidget(title, 1)

        toggle_button = QPushButton("展开")
        toggle_button.setFixedWidth(58)
        header.addWidget(toggle_button, 0, Qt.AlignTop)
        card_layout.addLayout(header)

        detail = QLabel(_evidence_detail_text(option, by_id=by_id))
        detail.setWordWrap(True)
        detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail.setVisible(False)
        detail.setStyleSheet(
            "background:#f8fafc;border:1px solid #e1e7ef;border-radius:6px;padding:8px;line-height:1.45;"
        )
        card_layout.addWidget(detail)

        def toggle_detail() -> None:
            visible = not detail.isVisible()
            detail.setVisible(visible)
            toggle_button.setText("收起" if visible else "展开")

        toggle_button.clicked.connect(toggle_detail)
        self.followup_layout.insertWidget(max(0, self.followup_layout.count() - 1), card)
        self.evidence_cards.append({"checkbox": checkbox, "payload": option, "card": card})

    def _refresh_pinned_list(self) -> None:
        self.pinned_cards.clear()
        if hasattr(self, "pinned_layout"):
            while self.pinned_layout.count() > 1:
                item = self.pinned_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            by_id = {
                str(payload.get("option_id", "")): payload
                for payload in self.pinned_contexts
                if str(payload.get("option_id", "")).strip()
            }
            for payload in self.pinned_contexts:
                self._add_context_card(
                    payload,
                    by_id=by_id,
                    target_layout=self.pinned_layout,
                    registry=self.pinned_cards,
                    checkbox_tooltip="勾选后可从固定证据中移除",
                )
        self.pinned_summary_label.setText(f"已固定证据 {len(self.pinned_contexts)} 条")
        return
        self.pinned_list.clear()
        for payload in self.pinned_contexts:
            label = payload.get("label", "未命名证据")
            preview = payload.get("preview", "")
            item = QListWidgetItem(f"{label}\n{preview}")
            item.setData(Qt.UserRole, payload.get("option_id"))
            self.pinned_list.addItem(item)
        self.pinned_summary_label.setText(f"已固定证据 {len(self.pinned_contexts)} 条")

    def _render_answer_details(self, retrieval_result, validated, *, selected_contexts: list[dict] | None = None) -> None:
        selected_context_text = format_selected_contexts(
            self._selected_context_payloads() if selected_contexts is None else selected_contexts
        )
        selected_context_text = _truncate_ui_text(selected_context_text, 12000)
        retrieved_context_text = _truncate_ui_text(retrieval_result.context_text or "No usable chunks", 24000)
        parts = [
            f"<b>Knowledge Related:</b> {html.escape(str(validated.is_related))}",
            f"<b>Blocked:</b> {html.escape(str(validated.is_blocked))}",
            f"<b>Premise Status:</b> {html.escape(validated.premise_status)}",
            f"<b>Answer Mode:</b> {html.escape(validated.answer_mode)}",
            f"<b>Reason:</b> {html.escape(validated.reason)}",
            f"<b>Evidence Quotes:</b> {html.escape(' | '.join(validated.evidence_quotes) if validated.evidence_quotes else '无')}",
            f"<b>Search Scope:</b> {html.escape(_format_search_scope(self.pending_search_scope or self._selected_search_scope()))}",
            f"<b>Core Question:</b> {html.escape(retrieval_result.query_plan.core_question)}",
            f"<b>Retrieval Focus:</b> {html.escape(retrieval_result.query_plan.retrieval_focus)}",
            f"<b>Retrieval Query:</b> {html.escape(retrieval_result.retrieval_query)}",
            f"<b>LLM Query Preprocessing:</b> {html.escape('used' if getattr(retrieval_result.query_plan, 'used_llm_enrichment', False) else 'heuristic fallback')}",
            f"<b>Query Modes:</b> {html.escape(', '.join(retrieval_result.query_plan.query_modes))}",
            f"<b>Keywords:</b> {html.escape(', '.join(retrieval_result.keywords))}",
            f"<b>Pinned / Selected Follow-up Evidence:</b><br><pre>{html.escape(selected_context_text or '无')}</pre>",
            f"<b>Retrieved Context:</b><br><pre>{html.escape(retrieved_context_text)}</pre>",
        ]
        self.details_browser.setHtml("<br>".join(parts))

    def _render_debug_table(self, rows: list[dict]) -> None:
        if not rows:
            self.debug_table.clear()
            self.debug_table.setRowCount(0)
            self.debug_table.setColumnCount(0)
            return
        headers = list(rows[0].keys())
        self.debug_table.setColumnCount(len(headers))
        self.debug_table.setHorizontalHeaderLabels(headers)
        self.debug_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, header in enumerate(headers):
                value = _compact_table_value(row.get(header, ""))
                self.debug_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        for col_index in range(len(headers)):
            self.debug_table.setColumnWidth(col_index, 140)

    def _append_chat(self, role: str, text: str, *, blocked: bool = False) -> None:
        self.chat_message_index += 1
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)

        bubble = QFrame()
        bubble.setProperty("chatBubbleUser" if role == "user" else "chatBubbleAssistant", True)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 9, 12, 10)
        bubble_layout.setSpacing(6)

        role_label = QLabel("You" if role == "user" else "Assistant")
        role_label.setProperty("chatRole", True)
        bubble_layout.addWidget(role_label)

        content = QLabel(text)
        content.setWordWrap(True)
        content.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content.setMinimumWidth(180)
        content.setMaximumWidth(520)
        bubble_layout.addWidget(content)

        if role == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0, Qt.AlignRight)
        else:
            row_layout.addWidget(bubble, 0, Qt.AlignLeft)
            row_layout.addStretch(1)

        self.chat_layout.insertWidget(max(0, self.chat_layout.count() - 1), row)
        self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum())
        return
        bubble_color = "#eaf4ff" if role == "user" else ("#fff7e8" if blocked else "#f4f7fb")
        role_label = "用户" if role == "user" else "助手"
        block = (
            f"<div style='margin:10px 0;padding:12px 14px;border-radius:8px;"
            f"background:{bubble_color};border:1px solid #dbe3ee;'>"
            f"<div style='font-weight:600;margin-bottom:6px;'>{html.escape(role_label)}</div>"
            f"<div style='white-space:pre-wrap;'>{html.escape(text)}</div>"
            f"</div>"
        )
        self.chat_browser.append(block)

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _selected_corpus_names(self) -> list[str]:
        return list(self._selected_search_scope().get("corpora", []))

    def _selected_search_scope(self) -> dict[str, Any]:
        selected: list[str] = []
        volumes: dict[str, list[int]] = {}
        if not hasattr(self, "corpus_tree"):
            return {"corpora": selected, "volumes": volumes}
        for row in range(self.corpus_tree.topLevelItemCount()):
            root = self.corpus_tree.topLevelItem(row)
            data = root.data(0, Qt.UserRole) or {}
            corpus_name = str(data.get("corpus_name") or root.text(0)).strip()
            if not corpus_name or root.checkState(0) == Qt.Unchecked:
                continue
            selected.append(corpus_name)
            child_indices: list[int] = []
            checked_children = 0
            checkable_children = 0
            for index in range(root.childCount()):
                child = root.child(index)
                child_data = child.data(0, Qt.UserRole) or {}
                if child_data.get("type") != "volume":
                    continue
                checkable_children += 1
                if child.checkState(0) == Qt.Checked:
                    checked_children += 1
                    try:
                        child_indices.append(int(child_data.get("volume_index")))
                    except (TypeError, ValueError):
                        logger.debug("分卷索引无法解析为整数,已跳过:%r", child_data.get("volume_index"))
            if checkable_children and checked_children and checked_children < checkable_children:
                volumes[corpus_name] = sorted(set(child_indices))
        return {"corpora": selected, "volumes": volumes}

    def _populate_model_combo(self, combo: DropdownSelect) -> None:
        options = [
            ("Pro", "dsv4pro"),
            ("Flash", "dsv4flash"),
        ]
        for label, value in options:
            combo.addItem(label, value)
        combo.setCurrentIndex(0)

    def _selected_ask_model(self) -> str:
        return str(self.ask_model_combo.currentData() or "dsv4pro")

    def _selected_build_model(self) -> str:
        return str(self.build_model_combo.currentData() or "dsv4pro")

    def _selected_context_payloads(self) -> list[dict]:
        selected = list(self.pinned_contexts)
        known_ids = {str(item.get("option_id", "")) for item in selected}
        for card in self.evidence_cards:
            checkbox = card.get("checkbox")
            if checkbox is None or not checkbox.isChecked():
                continue
            payload = card.get("payload") or {}
            option_id = str(payload.get("option_id", ""))
            if option_id and option_id in known_ids:
                continue
            selected.append(payload)
            if option_id:
                known_ids.add(option_id)
        return selected

    def _set_all_corpora_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        self.corpus_tree.blockSignals(True)
        for row in range(self.corpus_tree.topLevelItemCount()):
            root = self.corpus_tree.topLevelItem(row)
            root.setCheckState(0, state)
            for index in range(root.childCount()):
                child = root.child(index)
                data = child.data(0, Qt.UserRole) or {}
                if data.get("type") == "volume":
                    child.setCheckState(0, state)
        self.corpus_tree.blockSignals(False)
        self._update_scope_warning()

    def _handle_scope_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(0, Qt.UserRole) or {}
        self.corpus_tree.blockSignals(True)
        if data.get("type") == "corpus":
            state = item.checkState(0)
            for index in range(item.childCount()):
                child = item.child(index)
                child_data = child.data(0, Qt.UserRole) or {}
                if child_data.get("type") == "volume":
                    child.setCheckState(0, state)
        elif data.get("type") == "volume":
            parent = item.parent()
            if parent is not None:
                any_checked = any(
                    parent.child(index).checkState(0) == Qt.Checked
                    for index in range(parent.childCount())
                    if (parent.child(index).data(0, Qt.UserRole) or {}).get("type") == "volume"
                )
                parent.setCheckState(0, Qt.Checked if any_checked else Qt.Unchecked)
        self.corpus_tree.blockSignals(False)
        self._update_scope_warning()

    def _update_scope_warning(self) -> None:
        scope = self._selected_search_scope()
        has_volume_filter = any(values for values in scope.get("volumes", {}).values())
        if has_volume_filter:
            self.scope_warning_label.setText("已选择具体卷：未标卷旧内容不会参与搜索；如需纳入请重建知识库。")
        else:
            self.scope_warning_label.setText("")
        self._update_question_placeholder()

    def _update_question_placeholder(self) -> None:
        if not hasattr(self, "question_input"):
            return
        scope = self._selected_search_scope() if hasattr(self, "corpus_tree") else {"corpora": []}
        corpora = list(scope.get("corpora", []) or [])
        if not self.has_corpora:
            self.question_input.setPlaceholderText("需要先建库，点击这里选择 TXT 并建库")
            return
        if not corpora:
            self.question_input.setPlaceholderText("请先在左侧选择要搜索的知识库")
            return
        names = corpora[:2]
        joined = " 和 ".join(names)
        if len(corpora) > 2:
            joined += f" 等 {len(corpora)} 个知识库"
        self.question_input.setPlaceholderText(f"围绕 {joined} 提问")

    def clear_pinned_button_proxy(self) -> QPushButton:
        button = QPushButton("清空全部")
        button.clicked.connect(self.clear_pinned_contexts)
        return button

    @staticmethod
    def _card() -> QFrame:
        frame = QFrame()
        frame.setProperty("card", True)
        return frame

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("sectionTitle", True)
        return label


def _truncate_ui_text(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[UI truncated: showing first {limit} characters]"


def _compact_table_value(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_search_scope(scope: dict[str, Any]) -> str:
    corpora = list(scope.get("corpora", []) or [])
    volumes = scope.get("volumes", {}) if isinstance(scope.get("volumes", {}), dict) else {}
    if not corpora:
        return "未选择知识库"
    parts = []
    for corpus in corpora:
        selected_volumes = volumes.get(corpus) or []
        if selected_volumes:
            rendered = "、".join(f"第{int(index)}卷" for index in selected_volumes)
            parts.append(f"{corpus}（{rendered}）")
        else:
            parts.append(f"{corpus}（全部）")
    return "；".join(parts)


def _evidence_source_label(option: dict[str, Any]) -> str:
    role = str(option.get("context_role_label") or option.get("context_role") or "Evidence")
    source = str(option.get("source") or "").strip()
    chapter = str(option.get("chapter") or "").strip()
    chunk_index = str(option.get("chunk_index") or "?").strip()
    score = option.get("score")
    score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "?"
    if option.get("context_merge_reason") == "continuous_retrieval_chunks":
        prefix = "[Merged]"
    else:
        prefix = "[Expanded]" if option.get("has_expanded_context") else ("[Neighbor]" if option.get("is_context_expansion") else "[Primary]")
    return f"{prefix} {source} / {chapter} / chunk {chunk_index} / score {score_text} / {role}"


def _evidence_detail_text(option: dict[str, Any], *, by_id: dict[str, dict] | None = None) -> str:
    parts = []
    persons = ", ".join(str(value) for value in option.get("persons", [])[:8] if value)
    events = ", ".join(str(value) for value in option.get("events", [])[:6] if value)
    reason = str(option.get("expansion_reason") or "").strip()
    if persons:
        parts.append(f"Persons: {persons}")
    if events:
        parts.append(f"Events: {events}")
    if reason:
        parts.append(f"Expansion reason: {reason}")
    context_parts = _ordered_evidence_context(option, by_id or {})
    if context_parts:
        parts.append("\n\n".join(context_parts))
    return "\n\n".join(parts) if parts else "(No preview text)"


def _ordered_evidence_context(option: dict[str, Any], by_id: dict[str, dict]) -> list[str]:
    current_text = str(option.get("page_content") or option.get("preview") or "").strip()
    if not current_text:
        return []

    if option.get("has_expanded_context"):
        return [f"【合并上下文】\n{current_text}"]

    if option.get("is_context_expansion"):
        return [f"【相邻上下文】\n{current_text}"]

    chunks: list[str] = []
    prev_id = str(option.get("prev_chunk_id") or "").strip()
    next_id = str(option.get("next_chunk_id") or "").strip()
    prev = by_id.get(prev_id)
    next_item = by_id.get(next_id)

    if prev:
        prev_text = str(prev.get("page_content") or "").strip()
        if prev_text:
            chunks.append(f"【前一条】\n{prev_text}")
    elif prev_id:
        chunks.append("【前一条】\n未进入本轮召回结果。")

    chunks.append(f"【本条】\n{current_text}")

    if next_item:
        next_text = str(next_item.get("page_content") or "").strip()
        if next_text:
            chunks.append(f"【后一条】\n{next_text}")
    elif next_id:
        chunks.append("【后一条】\n未进入本轮召回结果。")

    return chunks
