"""统一日志工具。

集中配置 logging,替代项目中散落的 print 与静默 except: pass。
其它模块统一通过 ``get_logger(__name__)`` 获取 logger,避免各自重复配置。

日志级别可用环境变量 ``RAG_LOG_LEVEL`` 控制(默认 INFO),例如:
    RAG_LOG_LEVEL=DEBUG
"""

from __future__ import annotations

import logging
import os

# 全项目共用的根 logger 名称,子 logger 以 "storyrag.xxx" 形式挂在其下
_ROOT_NAME = "storyrag"
_configured = False


def _configure_root() -> None:
    """惰性配置根 logger,保证只初始化一次(幂等)。"""
    global _configured
    if _configured:
        return

    level_name = os.getenv("RAG_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    # 避免重复添加 handler(例如多次 import 或测试重载时)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    # 由本项目根 logger 自行输出,不再向 Python 全局 root 传播造成重复打印
    root.propagate = False
    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """返回项目统一命名空间下的 logger。

    参数 name 通常传入调用方的 ``__name__``;为简洁起见会自动挂到
    ``storyrag`` 命名空间下。
    """
    _configure_root()
    if not name or name == "__main__":
        return logging.getLogger(_ROOT_NAME)
    # 统一收敛到 storyrag.<模块名> 命名空间
    short = name.split(".")[-1]
    return logging.getLogger(f"{_ROOT_NAME}.{short}")
