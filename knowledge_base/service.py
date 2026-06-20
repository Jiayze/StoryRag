from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from env_loader import load_project_env
from core.config import DOC_DIR
from preprocessing import (
    ChunkArtifact,
    PreprocessingResult,
    chunk_to_payload,
    ensure_processed_dir,
    load_corpus_result,
    persist_corpus_result,
    persist_preprocessing_result,
    preprocess_files,
)
from preprocessing.storage import PROCESSED_DIR
from retrieval import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    COLLECTION_CONFIGURATION,
    COLLECTION_METADATA,
    get_embedding_model,
    load_vector_db,
)


load_project_env()



def ensure_doc_dir(directory: Path = DOC_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_corpus_dir(corpus_name: str, directory: Path = DOC_DIR) -> Path:
    corpus_dir = ensure_doc_dir(directory) / corpus_name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    return corpus_dir


def get_all_txt_files(directory: Path = DOC_DIR) -> list[Path]:
    ensure_doc_dir(directory)
    return sorted(path for path in directory.rglob("*.txt") if path.is_file())


def get_corpus_txt_files(corpus_name: str, directory: Path = DOC_DIR) -> list[Path]:
    corpus_dir = ensure_corpus_dir(corpus_name, directory)
    return sorted(path for path in corpus_dir.rglob("*.txt") if path.is_file())


def list_available_corpora() -> list[str]:
    names = []
    doc_dir = ensure_doc_dir(DOC_DIR)
    for entry in sorted(doc_dir.iterdir()):
        if entry.is_dir():
            names.append(entry.name)
    return names


def build_vector_db_from_files(
    file_paths: list[Path],
    db_path: str = CHROMA_DB_DIR,
    *,
    use_llm_enrichment: bool | None = None,
    llm_model: str | None = None,
    incremental: bool = False,
) -> tuple[Chroma, list[Document]]:
    if not file_paths:
        raise ValueError("No .txt files were provided for indexing.")

    doc_dir = ensure_doc_dir(DOC_DIR)
    normalized_paths = [Path(path).resolve() for path in file_paths]
    result = preprocess_files(
        normalized_paths,
        base_dir=doc_dir,
        use_llm_enrichment=use_llm_enrichment,
        llm_model=llm_model,
    )
    if not result.chunks:
        raise ValueError("No valid text chunks were generated from the provided files.")

    ensure_processed_dir(PROCESSED_DIR)
    persist_preprocessing_result(result, output_dir=PROCESSED_DIR)

    documents = [artifact_to_document(chunk) for chunk in result.chunks]
    if incremental:
        vector_db = upsert_documents(documents, db_path=db_path)
    else:
        vector_db = rebuild_vector_db(documents, db_path=db_path)
    return vector_db, documents


def build_corpus_vector_db(
    corpus_name: str,
    *,
    file_paths: list[Path] | None = None,
    use_llm_enrichment: bool | None = None,
    llm_model: str | None = None,
    db_path: str = CHROMA_DB_DIR,
) -> tuple[Chroma, list[Document]]:
    target_files = file_paths or get_corpus_txt_files(corpus_name, DOC_DIR)
    if not target_files:
        raise ValueError(f"No .txt files found for corpus '{corpus_name}'.")

    doc_dir = ensure_doc_dir(DOC_DIR)
    normalized_paths = [Path(path).resolve() for path in target_files]
    result = preprocess_files(
        normalized_paths,
        base_dir=doc_dir,
        use_llm_enrichment=use_llm_enrichment,
        llm_model=llm_model,
    )
    if not result.chunks:
        raise ValueError("No valid text chunks were generated from the provided files.")

    ensure_processed_dir(PROCESSED_DIR)
    persist_preprocessing_result(result, output_dir=PROCESSED_DIR)

    documents = [artifact_to_document(chunk) for chunk in result.chunks]
    vector_db = _load_or_create_vector_db(Path(db_path))
    _delete_existing_corpus_documents(vector_db, corpus_name)
    _add_documents(vector_db, documents)
    return vector_db, documents


def update_corpus_vector_db_incremental(
    corpus_name: str,
    *,
    file_paths: list[Path],
    use_llm_enrichment: bool | None = None,
    llm_model: str | None = None,
    db_path: str = CHROMA_DB_DIR,
) -> tuple[Chroma, list[Document], dict[str, Any]]:
    if not file_paths:
        raise ValueError("No .txt files were provided for incremental indexing.")

    doc_dir = ensure_doc_dir(DOC_DIR)
    normalized_paths = [Path(path).resolve() for path in file_paths]
    existing = load_corpus_result(corpus_name, PROCESSED_DIR)
    existing_by_relative = {
        str(document.relative_path).replace("\\", "/"): document
        for document in (existing.documents if existing else [])
    }

    changed_paths: list[Path] = []
    skipped_paths: list[Path] = []
    for path in normalized_paths:
        loaded = _quick_load_text(path, doc_dir)
        existing_doc = existing_by_relative.get(loaded["relative_path"])
        if existing_doc and existing_doc.normalized_sha1 == loaded["normalized_sha1"]:
            skipped_paths.append(path)
        else:
            changed_paths.append(path)

    vector_db = _load_or_create_vector_db(Path(db_path))
    if not changed_paths:
        return vector_db, [], {
            "added_files": 0,
            "updated_files": 0,
            "skipped_files": len(skipped_paths),
            "new_chunks": 0,
            "role_index_rebuilt": False,
        }

    result = preprocess_files(
        changed_paths,
        base_dir=doc_dir,
        use_llm_enrichment=use_llm_enrichment,
        llm_model=llm_model,
    )
    if not result.chunks:
        raise ValueError("No valid text chunks were generated from the selected files.")

    changed_doc_ids = {document.doc_id for document in result.documents}
    changed_relative_paths = {document.relative_path for document in result.documents}
    existing_replaced_docs = [
        document
        for document in (existing.documents if existing else [])
        if document.relative_path in changed_relative_paths
    ]
    replaced_doc_ids = {document.doc_id for document in existing_replaced_docs}
    ids_to_delete = _chunk_ids_for_docs(existing, replaced_doc_ids) if existing else []
    ids_to_delete.extend(_role_index_chunk_ids(existing) if existing else [])
    if ids_to_delete:
        print(f"[INFO] Incremental update removes {len(ids_to_delete)} stale chunks before upsert.")
        vector_db.delete(ids=sorted(set(ids_to_delete)))

    merged_result = _merge_corpus_results(
        corpus_name=corpus_name,
        existing=existing,
        incoming=result,
        replaced_doc_ids=replaced_doc_ids,
        changed_doc_ids=changed_doc_ids,
    )
    merged_result = _rebuild_corpus_role_index_from_existing(corpus_name, merged_result)
    ensure_processed_dir(PROCESSED_DIR)
    persist_corpus_result(corpus_name, merged_result, output_dir=PROCESSED_DIR)

    new_doc_ids = changed_doc_ids | _role_index_doc_ids(merged_result)
    documents = [
        artifact_to_document(chunk)
        for chunk in merged_result.chunks
        if chunk.doc_id in changed_doc_ids or chunk.metadata.get("is_synthetic_role_index")
    ]
    _add_documents(vector_db, documents)
    added_files = sum(1 for document in result.documents if document.doc_id not in replaced_doc_ids)
    updated_files = len(result.documents) - added_files
    return vector_db, documents, {
        "added_files": added_files,
        "updated_files": updated_files,
        "skipped_files": len(skipped_paths),
        "new_chunks": len([chunk for chunk in result.chunks if not chunk.metadata.get("is_synthetic_role_index")]),
        "written_chunks": len(documents),
        "role_index_rebuilt": True,
    }


def rebuild_existing_corpus(
    corpus_name: str,
    *,
    use_llm_enrichment: bool | None = None,
    llm_model: str | None = None,
    db_path: str = CHROMA_DB_DIR,
) -> tuple[Chroma, list[Document]]:
    target_files = get_corpus_txt_files(corpus_name, DOC_DIR)
    if not target_files:
        raise ValueError(f"No .txt files found for corpus '{corpus_name}'.")
    return build_corpus_vector_db(
        corpus_name,
        file_paths=target_files,
        use_llm_enrichment=use_llm_enrichment,
        llm_model=llm_model,
        db_path=db_path,
    )


def _quick_load_text(path: Path, base_dir: Path) -> dict[str, str]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    raw_text = None
    for encoding in encodings:
        try:
            raw_text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if raw_text is None:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    try:
        relative_path = str(path.relative_to(base_dir)).replace("\\", "/")
    except Exception:
        relative_path = path.name
    return {
        "relative_path": relative_path,
        "normalized_sha1": _sha1_hex(normalized),
        "char_count": str(len(normalized)),
    }


def _merge_corpus_results(
    *,
    corpus_name: str,
    existing: PreprocessingResult | None,
    incoming: PreprocessingResult,
    replaced_doc_ids: set[str],
    changed_doc_ids: set[str],
) -> PreprocessingResult:
    if existing is None:
        return incoming
    kept_doc_ids = {
        document.doc_id
        for document in existing.documents
        if document.doc_id not in replaced_doc_ids
    }
    documents = [document for document in existing.documents if document.doc_id in kept_doc_ids]
    chapters = [chapter for chapter in existing.chapters if chapter.doc_id in kept_doc_ids]
    chunks = [
        chunk
        for chunk in existing.chunks
        if chunk.doc_id in kept_doc_ids and not chunk.metadata.get("is_synthetic_role_index")
    ]
    relations = [relation for relation in existing.relations if relation.doc_id in kept_doc_ids]
    documents.extend(incoming.documents)
    chapters.extend(incoming.chapters)
    chunks.extend(chunk for chunk in incoming.chunks if not chunk.metadata.get("is_synthetic_role_index"))
    relations.extend(incoming.relations)
    return PreprocessingResult(
        pipeline_version=incoming.pipeline_version,
        generated_at=incoming.generated_at,
        documents=documents,
        chapters=chapters,
        chunks=chunks,
        relations=relations,
    )


def _rebuild_corpus_role_index_from_existing(corpus_name: str, result: PreprocessingResult) -> PreprocessingResult:
    source_chunks = [chunk for chunk in result.chunks if not chunk.metadata.get("is_synthetic_role_index")]
    if not source_chunks:
        return result
    max_index = max((chunk.chunk_index for chunk in source_chunks), default=0) + 1
    role_chunks: list[ChunkArtifact] = []
    role_chunks.append(_make_role_index_chunk(corpus_name, "全文", None, source_chunks, max_index))
    max_index += 1
    by_volume: dict[tuple[str, int], list[ChunkArtifact]] = {}
    for chunk in source_chunks:
        label = str(chunk.metadata.get("volume_label") or "").strip()
        index = chunk.metadata.get("volume_index")
        if not label or index is None:
            continue
        try:
            key = (label, int(index))
        except Exception:
            continue
        by_volume.setdefault(key, []).append(chunk)
    for (label, index), chunks in sorted(by_volume.items(), key=lambda item: item[0][1]):
        role_chunks.append(_make_role_index_chunk(corpus_name, label, index, chunks, max_index))
        max_index += 1
    result.chunks = source_chunks + [chunk for chunk in role_chunks if chunk is not None]
    return result


def _make_role_index_chunk(
    corpus_name: str,
    scope_label: str,
    volume_index: int | None,
    source_chunks: list[ChunkArtifact],
    chunk_index: int,
) -> ChunkArtifact:
    person_counter: dict[str, int] = {}
    relationship_counter: dict[str, int] = {}
    female_counter: dict[str, int] = {}
    male_counter: dict[str, int] = {}
    for chunk in source_chunks:
        for person in chunk.metadata.get("persons", []) or []:
            person_counter[str(person)] = person_counter.get(str(person), 0) + 1
        for relationship in chunk.metadata.get("relations", []) or chunk.metadata.get("important_relationships", []) or []:
            relationship_counter[str(relationship)] = relationship_counter.get(str(relationship), 0) + 1
        for person in chunk.metadata.get("female_characters", []) or []:
            female_counter[str(person)] = female_counter.get(str(person), 0) + 1
        for person in chunk.metadata.get("male_characters", []) or []:
            male_counter[str(person)] = male_counter.get(str(person), 0) + 1
    major = [name for name, _ in sorted(person_counter.items(), key=lambda item: item[1], reverse=True)[:16]]
    relationships = [name for name, _ in sorted(relationship_counter.items(), key=lambda item: item[1], reverse=True)[:12]]
    female = [name for name, _ in sorted(female_counter.items(), key=lambda item: item[1], reverse=True)[:10]]
    male = [name for name, _ in sorted(male_counter.items(), key=lambda item: item[1], reverse=True)[:10]]
    title = "角色总表" if volume_index is None else f"{scope_label}角色总表"
    text = "\n".join(
        part
        for part in [
            f"【{scope_label}角色总表】",
            f"主要角色：{'、'.join(major) if major else '暂无'}",
            f"女性角色：{'、'.join(female) if female else '暂无'}",
            f"男性角色：{'、'.join(male) if male else '暂无'}",
            f"重要关系：{'；'.join(relationships) if relationships else '暂无'}",
            "质量标记：基于已有 chunk metadata 汇总生成。",
        ]
        if part
    )
    doc_id = _stable_id("doc", corpus_name, "synthetic-role-index")
    chapter_id = _stable_id("chapter", doc_id, scope_label, "角色总表")
    chunk_id = _stable_id("chunk", doc_id, chapter_id, "role_index", scope_label, _sha1_hex(text))
    metadata: dict[str, Any] = {
        "persons": major[:12],
        "chapter_persons": major[:12],
        "keywords": ["角色总表", "人物关系", *major[:8]],
        "chapter_keywords": ["角色总表", "人物关系", *major[:8]],
        "locations": [],
        "events": [],
        "objects": [],
        "aliases": [],
        "chapter_locations": [],
        "chapter_events": [],
        "chapter_objects": [],
        "chapter_aliases": [],
        "relative_path": f"{corpus_name}/__role_index__",
        "is_synthetic_role_index": True,
        "role_index_scope": scope_label,
        "female_characters": female,
        "male_characters": male,
        "important_relationships": relationships,
        "source_chunk_ids": [chunk.chunk_id for chunk in source_chunks[:24]],
        "role_index_quality": "metadata_summary",
    }
    if volume_index is not None:
        metadata["volume_label"] = scope_label
        metadata["volume_index"] = volume_index
    return ChunkArtifact(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chapter_id=chapter_id,
        source_path=f"{corpus_name}/__role_index__",
        doc_name="角色总表",
        corpus_name=corpus_name,
        chapter_title=title,
        chapter_index=-1,
        chunk_index=chunk_index,
        char_start=0,
        char_end=0,
        text=text,
        summary=text,
        keywords=metadata["keywords"],
        metadata=metadata,
    )


def _chunk_ids_for_docs(result: PreprocessingResult | None, doc_ids: set[str]) -> list[str]:
    if result is None or not doc_ids:
        return []
    return [chunk.chunk_id for chunk in result.chunks if chunk.doc_id in doc_ids]


def _role_index_chunk_ids(result: PreprocessingResult | None) -> list[str]:
    if result is None:
        return []
    return [chunk.chunk_id for chunk in result.chunks if chunk.metadata.get("is_synthetic_role_index")]


def _role_index_doc_ids(result: PreprocessingResult) -> set[str]:
    return {chunk.doc_id for chunk in result.chunks if chunk.metadata.get("is_synthetic_role_index")}


def _stable_id(prefix: str, *parts: str) -> str:
    import hashlib

    joined = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(joined.encode('utf-8')).hexdigest()[:24]}"


def _sha1_hex(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def artifact_to_document(chunk) -> Document:
    chunk_id, metadata, text = chunk_to_payload(chunk)
    return Document(page_content=text, metadata=_sanitize_metadata_for_chroma(metadata), id=chunk_id)


def rebuild_vector_db(documents: list[Document], db_path: str = CHROMA_DB_DIR) -> Chroma:
    if not documents:
        raise ValueError("Cannot rebuild Chroma DB with empty documents.")

    persist_path = Path(db_path)
    temp_path = persist_path.with_name(f"{persist_path.name}_tmp")
    backup_path = persist_path.with_name(f"{persist_path.name}_bak")

    if os.name == "nt":
        return _rebuild_vector_db_in_place(documents, persist_path)

    if temp_path.exists():
        shutil.rmtree(temp_path)
    if backup_path.exists():
        shutil.rmtree(backup_path)

    print("[INFO] Initializing embedding model and writing documents to temporary Chroma DB.")
    embeddings_model = get_embedding_model()
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        ids=[str(doc.id or doc.metadata.get("chunk_id")) for doc in documents],
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(temp_path),
        collection_metadata=COLLECTION_METADATA,
        collection_configuration=COLLECTION_CONFIGURATION,
    )

    count = int(vector_db._collection.count())
    if count != len(documents):
        raise RuntimeError(
            f"Temporary Chroma collection count mismatch: expected {len(documents)}, got {count}."
        )

    del vector_db
    time.sleep(0.5)

    if persist_path.exists():
        persist_path.rename(backup_path)

    rename_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            temp_path.rename(persist_path)
        except PermissionError as exc:
            rename_error = exc
            print(f"[INFO] Waiting for Chroma files to unlock before swap (attempt {attempt}/5).")
            time.sleep(1.0 * attempt)
        except Exception:
            if backup_path.exists() and not persist_path.exists():
                backup_path.rename(persist_path)
            raise
        else:
            rename_error = None
            break

    if rename_error is not None:
        if backup_path.exists() and not persist_path.exists():
            backup_path.rename(persist_path)
        raise rename_error

    if backup_path.exists():
        shutil.rmtree(backup_path)

    return load_vector_db(str(persist_path))


def upsert_documents(documents: list[Document], db_path: str = CHROMA_DB_DIR) -> Chroma:
    if not documents:
        raise ValueError("Cannot upsert empty documents.")
    persist_path = Path(db_path)
    vector_db = _load_or_create_vector_db(persist_path)
    _add_documents(vector_db, documents)
    return vector_db


def _load_or_create_vector_db(persist_path: Path) -> Chroma:
    if persist_path.exists():
        return load_vector_db(str(persist_path))

    persist_path.mkdir(parents=True, exist_ok=True)
    embeddings_model = get_embedding_model()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(persist_path),
        embedding_function=embeddings_model,
        collection_metadata=COLLECTION_METADATA,
        collection_configuration=COLLECTION_CONFIGURATION,
    )


def _delete_existing_corpus_documents(vector_db: Chroma, corpus_name: str) -> None:
    try:
        existing = vector_db._collection.get(where={"corpus_name": corpus_name}, include=[])
    except Exception:
        return
    ids = existing.get("ids", []) if isinstance(existing, dict) else []
    if ids:
        print(f"[INFO] Removing {len(ids)} existing chunks for corpus '{corpus_name}'.")
        vector_db.delete(ids=ids)


def _add_documents(vector_db: Chroma, documents: list[Document]) -> None:
    ids = [str(doc.id or doc.metadata.get("chunk_id")) for doc in documents]
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    print(f"[INFO] Writing {len(documents)} chunks into Chroma.")
    vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)


def _rebuild_vector_db_in_place(documents: list[Document], persist_path: Path) -> Chroma:
    if persist_path.exists():
        print("[INFO] Windows mode: removing existing Chroma DB before in-place rebuild.")
        shutil.rmtree(persist_path)

    print("[INFO] Windows mode: rebuilding Chroma DB in place to avoid rename/file-lock issues.")
    embeddings_model = get_embedding_model()
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        ids=[str(doc.id or doc.metadata.get("chunk_id")) for doc in documents],
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(persist_path),
        collection_metadata=COLLECTION_METADATA,
        collection_configuration=COLLECTION_CONFIGURATION,
    )

    count = int(vector_db._collection.count())
    if count != len(documents):
        raise RuntimeError(
            f"Chroma collection count mismatch after in-place rebuild: expected {len(documents)}, got {count}."
        )

    print("[SUCCESS] Windows in-place Chroma rebuild completed.")
    return load_vector_db(str(persist_path))


def main() -> None:
    target_files = get_all_txt_files(DOC_DIR)

    if not target_files:
        print(f"[WARNING] No .txt files found in {DOC_DIR}.")
        print("[INFO] Put your source text files into the docs directory and run again.")
        return

    print(f"[INFO] Found {len(target_files)} text files in {DOC_DIR}:")
    for file_path in target_files:
        print(f" - {file_path}")

    print("\n[INFO] Start preprocessing documents and rebuilding vector DB...")
    vector_db, documents = build_vector_db_from_files(target_files)
    print(f"[SUCCESS] Generated {len(documents)} chunks.")
    print(f"[SUCCESS] Processed artifacts directory: {PROCESSED_DIR}")
    print(f"[SUCCESS] Chroma DB directory: {CHROMA_DB_DIR}")
    print(f"[SUCCESS] Collection count: {vector_db._collection.count()}")


def _sanitize_metadata_for_chroma(metadata: dict[str, object]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        cleaned = _sanitize_metadata_value(value)
        if cleaned is None:
            continue
        sanitized[key] = cleaned
    return sanitized


def _sanitize_metadata_value(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        if not value:
            return None
        scalar_items = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, (str, int, float, bool)):
                scalar_items.append(str(item) if not isinstance(item, bool) else ("true" if item else "false"))
            elif isinstance(item, dict):
                scalar_items.append(_flatten_dict_for_metadata(item))
            else:
                scalar_items.append(str(item))
        scalar_items = [item for item in scalar_items if str(item).strip()]
        if not scalar_items:
            return None
        return " | ".join(scalar_items)
    if isinstance(value, dict):
        if not value:
            return None
        return _flatten_dict_for_metadata(value)
    return str(value)


def _flatten_dict_for_metadata(value: dict[str, Any]) -> str:
    pairs = []
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, (str, int, float, bool)):
            rendered = item if isinstance(item, str) else str(item)
            if str(rendered).strip():
                pairs.append(f"{key}={rendered}")
        elif isinstance(item, list):
            if not item:
                continue
            rendered = ",".join(str(part) for part in item if str(part).strip())
            if rendered:
                pairs.append(f"{key}={rendered}")
        else:
            rendered = str(item).strip()
            if rendered:
                pairs.append(f"{key}={rendered}")
    return "; ".join(pairs) if pairs else ""
