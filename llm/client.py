from __future__ import annotations

import os

from openai import OpenAI

from env_loader import load_project_env


load_project_env()


def _normalize_model_name(raw: str | None) -> str:
    name = (raw or "deepseek-v4-pro").strip()
    aliases = {
        "dsv4pro": "deepseek-v4-pro",
        "ds-v4-pro": "deepseek-v4-pro",
        "deepseek-v4pro": "deepseek-v4-pro",
        "dsv4flash": "deepseek-v4-flash",
        "ds-v4-flash": "deepseek-v4-flash",
        "deepseek-v4flash": "deepseek-v4-flash",
    }
    return aliases.get(name.lower(), name)


DEEPSEEK_MODEL = _normalize_model_name(os.getenv("DEEPSEEK_MODEL"))


def _request_timeout() -> float:
    raw = os.getenv("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "45").strip()
    try:
        value = float(raw)
    except Exception:
        value = 45.0
    return max(5.0, value)


def normalize_deepseek_model(raw: str | None) -> str:
    return _normalize_model_name(raw)


def create_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY, cannot call the answer model.")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        timeout=_request_timeout(),
    )


def ensure_embedding_key() -> None:
    if not os.getenv("SILICONFLOW_API_KEY"):
        raise RuntimeError("Missing SILICONFLOW_API_KEY, cannot build the embedding index.")
