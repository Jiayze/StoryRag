"""核心基础设施包:统一日志、集中配置等。"""

from __future__ import annotations

from .logger import get_logger

__all__ = ["get_logger"]
