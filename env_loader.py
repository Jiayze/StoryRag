import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_env(filename: str = ".env", *, base_dir: Path | None = None) -> None:
    env_path = (base_dir or PROJECT_ROOT) / filename
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def resolve_project_path(value: str | Path | None, default: str | Path) -> Path:
    raw_value = default if value is None or str(value).strip() == "" else value
    path = Path(raw_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
