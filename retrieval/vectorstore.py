from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, COLLECTION_CONFIGURATION, COLLECTION_METADATA
from .embedding import get_embedding_model


def load_vector_db(persist_directory: str = CHROMA_DB_DIR) -> Chroma:
    embeddings_model = get_embedding_model()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=persist_directory,
        embedding_function=embeddings_model,
        collection_metadata=COLLECTION_METADATA,
        collection_configuration=COLLECTION_CONFIGURATION,
    )


def embed_query_once(db: Chroma, query: str) -> list[float]:
    embedding_function = getattr(db, "_embedding_function", None)
    if embedding_function is None:
        raise RuntimeError("Vector DB has no embedding function configured.")
    if hasattr(embedding_function, "embed_query"):
        return embedding_function.embed_query(query)
    if hasattr(embedding_function, "embed_documents"):
        return embedding_function.embed_documents([query])[0]
    raise RuntimeError("Embedding function does not support query embedding.")


def similarity_search_with_score_by_embedding(
    db: Chroma,
    embedding: list[float],
    *,
    k: int,
    filter: dict | None = None,
    where_document: dict | None = None,
) -> list[tuple[Document, float]]:
    return db.similarity_search_by_vector_with_relevance_scores(
        embedding,
        k=k,
        filter=filter,
        where_document=where_document,
    )


def similarity_search_by_embedding(
    db: Chroma,
    embedding: list[float],
    *,
    k: int,
    filter: dict | None = None,
    where_document: dict | None = None,
) -> list[Document]:
    return db.similarity_search_by_vector(
        embedding,
        k=k,
        filter=filter,
        where_document=where_document,
    )
