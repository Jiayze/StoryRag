from __future__ import annotations

import sys
import traceback

from PySide6.QtWidgets import QApplication

from env_loader import load_project_env

from .theme import apply_theme
from .window import StoryRagWindow


def _install_excepthook() -> None:
    def handle_exception(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = handle_exception


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    load_project_env()
    _install_excepthook()

    app = QApplication(argv)
    app.setApplicationName("StoryRAG Desktop")
    apply_theme(app)

    window = StoryRagWindow()
    if "--smoke-test" in argv:
        print("SMOKE_OK")
        return 0

    window.show()
    return app.exec()
