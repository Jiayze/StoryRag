from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    if not VENV_PYTHON.exists():
        raise SystemExit("Missing virtualenv python at .venv\\Scripts\\python.exe")

    env = dict(os.environ)
    command = [
        str(VENV_PYTHON),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name",
        "StoryRAGDesktop",
        "--collect-all",
        "PySide6",
        "--add-data",
        ".env;.",
        "run_desktop.py",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
