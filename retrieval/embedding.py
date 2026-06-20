from __future__ import annotations

import time

from openai import OpenAI

from .config import EMBEDDING_API_BASE, EMBEDDING_API_KEY, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


class CompatibleOpenAIEmbeddings:
    def __init__(
        self,
        *,
        model: str,
        openai_api_key: str,
        openai_api_base: str,
        chunk_size: int,
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self._last_heartbeat = time.monotonic()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        total = len(texts)
        print(f"[INFO] Embedding {total} text chunks with model={self.model}, batch_size={self.chunk_size}.")
        for batch_index, start in enumerate(range(0, len(texts), self.chunk_size), start=1):
            batch = texts[start : start + self.chunk_size]
            print(
                f"[INFO] Embedding batch {batch_index}: items {start + 1}-{start + len(batch)} / {total}."
            )
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
            now = time.monotonic()
            if now - self._last_heartbeat >= 30:
                print(
                    f"[INFO] Embedding heartbeat: completed {start + len(batch)} / {total} chunks."
                )
                self._last_heartbeat = now
        print(f"[SUCCESS] Embedding completed for {total} text chunks.")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding_model() -> CompatibleOpenAIEmbeddings:
    if not EMBEDDING_API_KEY:
        raise RuntimeError("Missing SILICONFLOW_API_KEY, cannot create embedding model.")
    return CompatibleOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=EMBEDDING_API_KEY,
        openai_api_base=EMBEDDING_API_BASE,
        chunk_size=EMBEDDING_BATCH_SIZE,
    )
