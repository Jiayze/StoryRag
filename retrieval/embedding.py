from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from core import get_logger

from .config import (
    EMBEDDING_API_BASE,
    EMBEDDING_API_KEY,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CONCURRENCY,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL,
    EMBEDDING_REQUEST_TIMEOUT,
)

logger = get_logger(__name__)

# 心跳最小间隔(秒):建库批次很多时,避免日志刷屏
_HEARTBEAT_INTERVAL = 30.0
# 重试退避基数(秒):第 n 次失败后等待 _BACKOFF_BASE * 2**(n-1)
_BACKOFF_BASE = 1.0


class CompatibleOpenAIEmbeddings:
    def __init__(
        self,
        *,
        model: str,
        openai_api_key: str,
        openai_api_base: str,
        chunk_size: int,
        timeout: float = EMBEDDING_REQUEST_TIMEOUT,
        concurrency: int = EMBEDDING_CONCURRENCY,
        max_retries: int = EMBEDDING_MAX_RETRIES,
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        # openai 客户端底层 httpx 连接池线程安全,可被多线程共享
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=timeout)
        self.concurrency = max(1, int(concurrency))
        self.max_retries = max(1, int(max_retries))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batches = [texts[start : start + self.chunk_size] for start in range(0, len(texts), self.chunk_size)]
        total = len(texts)
        n_batches = len(batches)
        workers = max(1, min(self.concurrency, n_batches))
        logger.info(
            "Embedding %d chunks in %d batches (model=%s, batch_size=%d, concurrency=%d).",
            total, n_batches, self.model, self.chunk_size, workers,
        )

        # 按批次索引回填,保证拼接顺序与输入一致(并发完成顺序是乱的)
        results: list[list[list[float]] | None] = [None] * n_batches
        completed_chunks = 0
        last_heartbeat = time.monotonic()

        def _record(batch_index: int, vectors: list[list[float]]) -> None:
            nonlocal completed_chunks, last_heartbeat
            results[batch_index] = vectors
            completed_chunks += len(vectors)
            now = time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                last_heartbeat = now
                logger.info("Embedding heartbeat: %d / %d chunks done.", completed_chunks, total)

        if workers == 1:
            for batch_index, batch in enumerate(batches):
                _record(batch_index, self._embed_batch_with_retry(batch, batch_index, n_batches))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_index = {
                    pool.submit(self._embed_batch_with_retry, batch, batch_index, n_batches): batch_index
                    for batch_index, batch in enumerate(batches)
                }
                # fut.result() 会把批次内重试耗尽后的异常重新抛出,让建库整体失败
                # 而非静默产出不完整向量(embedding 的完整性比"尽量跑完"更重要)。
                for fut in as_completed(future_to_index):
                    batch_index = future_to_index[fut]
                    _record(batch_index, fut.result())

        logger.info("Embedding completed for %d chunks.", total)
        embeddings: list[list[float]] = []
        for vectors in results:
            embeddings.extend(vectors or [])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        # 查询热路径:单批、不打建库日志,避免每次提问刷屏
        return self._embed_batch_with_retry([text], 0, 1, quiet=True)[0]

    def _embed_batch_with_retry(
        self,
        batch: list[str],
        batch_index: int,
        n_batches: int,
        *,
        quiet: bool = False,
    ) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                vectors = [item.embedding for item in response.data]
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"embedding count mismatch: got {len(vectors)} for {len(batch)} inputs"
                    )
                return vectors
            except Exception as exc:  # noqa: BLE001 — 需要对任意网络/解析异常重试
                last_exc = exc
                if attempt < self.max_retries:
                    sleep_s = _BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "Embedding batch %d/%d failed (attempt %d/%d): %s; retrying in %.1fs.",
                        batch_index + 1, n_batches, attempt, self.max_retries, exc, sleep_s,
                    )
                    time.sleep(sleep_s)
                elif not quiet:
                    logger.error(
                        "Embedding batch %d/%d failed after %d attempts: %s",
                        batch_index + 1, n_batches, self.max_retries, exc,
                    )
        assert last_exc is not None
        raise last_exc


def get_embedding_model() -> CompatibleOpenAIEmbeddings:
    if not EMBEDDING_API_KEY:
        raise RuntimeError("Missing SILICONFLOW_API_KEY, cannot create embedding model.")
    return CompatibleOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=EMBEDDING_API_KEY,
        openai_api_base=EMBEDDING_API_BASE,
        chunk_size=EMBEDDING_BATCH_SIZE,
    )
