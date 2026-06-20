"""集中配置:应用自身的全部 RAG_* 调参旋钮。

设计边界:
- 本模块只收口"应用自己的可调参数"(以 ``RAG_`` 前缀的数值/路径/开关),
  作为这些配置的**唯一声明处**。其它模块从这里导入(并按需再导出),
  不再各自散落调用 ``os.getenv``。
- **不**收口第三方 Provider 的凭证与模型名(``DEEPSEEK_API_KEY`` /
  ``DEEPSEEK_API_BASE`` / ``DEEPSEEK_MODEL`` / ``DEEPSEEK_REQUEST_TIMEOUT_SECONDS`` /
  ``SILICONFLOW_API_KEY`` / ``SILICONFLOW_API_BASE`` 以及
  ``RAG_PREPROCESS_DEEPSEEK_MODEL`` / ``RAG_QUERY_PREPROCESS_DEEPSEEK_MODEL``)。
  原因:它们要么按调用实时读取以支持"运行时改设置即时生效",要么与模型名
  归一化逻辑强耦合,集中到此反而会破坏现有行为。这些仍留在 ``llm/client.py`` 与
  ``preprocessing`` / ``retrieval`` 各自模块中。
- ``RAG_LOG_LEVEL`` 由 :mod:`core.logger` 自行读取,避免与日志初始化产生循环依赖。

所有默认值与各模块迁移前保持完全一致。
"""

from __future__ import annotations

import os
from pathlib import Path

from env_loader import load_project_env, resolve_project_path

# 引导:把 .env 装入 os.environ(setdefault 实现,幂等可重复调用)
load_project_env()


# --------------------------------------------------------------------------- #
# 带类型的环境变量读取助手(统一在此,避免各处手写 int()/float()/.lower())     #
# --------------------------------------------------------------------------- #
def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: str = "true") -> bool:
    # 与迁移前保持一致:取值小写后与 "true" 比较
    return os.getenv(name, default).lower() == "true"


def env_path(name: str, default: str) -> Path:
    return resolve_project_path(os.getenv(name), default)


# --------------------------------------------------------------------------- #
# 数据目录                                                                      #
# --------------------------------------------------------------------------- #
DOC_DIR = env_path("RAG_DOC_DIR", "docs")
PROCESSED_DIR = env_path("RAG_PROCESSED_DIR", "processed")
# 与历史保持一致:CHROMA_DB_DIR 对外是字符串
CHROMA_DB_DIR = str(env_path("RAG_CHROMA_DB_DIR", "chroma_db"))
PREPROCESS_CACHE_DIR = env_path("RAG_PREPROCESS_CACHE_DIR", "cache/preprocessing_llm")
QUERY_ENRICHMENT_CACHE_DIR = env_path(
    "RAG_QUERY_PREPROCESS_CACHE_DIR", "cache/query_preprocessing_llm"
)


# --------------------------------------------------------------------------- #
# 向量库 / Embedding(仅应用侧旋钮;SILICONFLOW 凭证仍在 retrieval/config.py)   #
# --------------------------------------------------------------------------- #
CHROMA_COLLECTION_NAME = env_str("RAG_CHROMA_COLLECTION", "langchain")
EMBEDDING_MODEL = env_str("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_BATCH_SIZE = env_int("RAG_EMBEDDING_BATCH_SIZE", 16)
# embedding(SiliconFlow)单次请求超时,防止某批次永久挂起
EMBEDDING_REQUEST_TIMEOUT = env_float("RAG_EMBEDDING_REQUEST_TIMEOUT_SECONDS", 60.0)
# embedding 批次并发数。实测 SiliconFlow 接口支持并发(线程共享 client 安全),
# 8 兼顾提速与 RPM/TPM 余量;长建库不建议 >16(会逼近 TPM 50 万/分 上限)。
EMBEDDING_CONCURRENCY = env_int("RAG_EMBEDDING_CONCURRENCY", 8)
# 单批 embedding 失败时的最大尝试次数(含首次),退避重试以兜住偶发 429/超时。
EMBEDDING_MAX_RETRIES = env_int("RAG_EMBEDDING_MAX_RETRIES", 3)
BGE_QUERY_PREFIX = env_str("RAG_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章：")


# --------------------------------------------------------------------------- #
# 检索:召回与排序                                                              #
# --------------------------------------------------------------------------- #
DEFAULT_TOP_K = env_int("RAG_TOP_K", 6)
DEFAULT_FETCH_K = env_int("RAG_FETCH_K", 40)
DEFAULT_KEYWORD_FETCH_K = env_int("RAG_KEYWORD_FETCH_K", 8)
DEFAULT_MAX_KEYWORDS = env_int("RAG_MAX_KEYWORDS", 6)
DEFAULT_DENSE_WEIGHT = env_float("RAG_DENSE_WEIGHT", 0.55)
DEFAULT_LEXICAL_WEIGHT = env_float("RAG_LEXICAL_WEIGHT", 0.20)
DEFAULT_METADATA_WEIGHT = env_float("RAG_METADATA_WEIGHT", 0.15)
DEFAULT_SUMMARY_WEIGHT = env_float("RAG_SUMMARY_WEIGHT", 0.10)
DEFAULT_RELATION_WEIGHT = env_float("RAG_RELATION_WEIGHT", 0.12)
DEFAULT_POSITION_WEIGHT = env_float("RAG_POSITION_WEIGHT", 0.0)
DEFAULT_MIN_HYBRID_SCORE = env_float("RAG_MIN_HYBRID_SCORE", 0.08)
DEFAULT_EXACT_KEYWORD_FIRST = env_bool("RAG_EXACT_KEYWORD_FIRST", "true")
# 原始字符串(可能为 None);派生出 float|None
MAX_DISTANCE = os.getenv("RAG_MAX_DISTANCE")
DEFAULT_MAX_DISTANCE = float(MAX_DISTANCE) if MAX_DISTANCE else None
MIN_KNOWN_PERSON_FREQUENCY = env_int("RAG_KNOWN_PERSON_MIN_FREQ", 3)


# --------------------------------------------------------------------------- #
# 预处理:分段与抽取                                                            #
# --------------------------------------------------------------------------- #
CHUNK_SIZE = env_int("RAG_PREPROCESS_CHUNK_SIZE", 800)
CHUNK_OVERLAP = env_int("RAG_PREPROCESS_CHUNK_OVERLAP", 120)
CHAPTER_SUMMARY_LENGTH = env_int("RAG_CHAPTER_SUMMARY_LENGTH", 220)
KEYWORD_LIMIT = env_int("RAG_PREPROCESS_KEYWORD_LIMIT", 12)
MIN_PERSON_FREQUENCY = env_int("RAG_MIN_PERSON_FREQUENCY", 2)
MIN_RELATION_PERSON_FREQUENCY = env_int("RAG_MIN_RELATION_PERSON_FREQUENCY", 2)


# --------------------------------------------------------------------------- #
# 预处理 / 查询:LLM 增强开关与字符上限(模型名仍在各自模块)                    #
# --------------------------------------------------------------------------- #
DEEPSEEK_PREPROCESS_ENABLED = env_bool("RAG_PREPROCESS_USE_DEEPSEEK", "true")
DEEPSEEK_CHAPTER_CHAR_LIMIT = env_int("RAG_PREPROCESS_CHAPTER_LLM_CHAR_LIMIT", 5000)
DEEPSEEK_CHUNK_CHAR_LIMIT = env_int("RAG_PREPROCESS_CHUNK_LLM_CHAR_LIMIT", 2200)
DEEPSEEK_ROLE_INDEX_CHAR_LIMIT = env_int("RAG_PREPROCESS_ROLE_INDEX_LLM_CHAR_LIMIT", 6000)
QUERY_ENRICHMENT_ENABLED = env_bool("RAG_QUERY_PREPROCESS_USE_DEEPSEEK", "true")
# 建库时每章 chunk LLM 增强的并发线程数(这些调用是纯网络等待,用线程池加速)。
# DeepSeek 并发上限很高(flash 2500 / pro 500),瓶颈在本地线程开销,32 兼顾速度与稳健。
PREPROCESS_CONCURRENCY = env_int("RAG_PREPROCESS_CONCURRENCY", 32)
# 单次预处理增强(DeepSeek)失败的最大尝试次数(含首次)。耗尽后降级为空增强
# (单章缺增强仍可建库,不应中断整次任务),但会记 error 而非静默吞掉。
PREPROCESS_MAX_RETRIES = env_int("RAG_PREPROCESS_MAX_RETRIES", 3)
