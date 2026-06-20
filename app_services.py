from __future__ import annotations

import json
import shutil
import os
import retrieval.config as retrieval_config
import retrieval.embedding as retrieval_embedding
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

from knowledge_base import (
    DOC_DIR,
    build_corpus_vector_db,
    ensure_corpus_dir,
    ensure_doc_dir,
    get_all_txt_files,
    get_corpus_txt_files,
    list_available_corpora,
    rebuild_existing_corpus,
    update_corpus_vector_db_incremental,
)
from llm import create_deepseek_client, ensure_embedding_key, normalize_deepseek_model
from preprocessing.storage import (
    CORPORA_DIRNAME,
    corpus_processed_dir,
    list_corpora,
    load_aliases,
    load_manifest,
    sanitize_corpus_name,
    save_aliases,
)
from qa import answer_with_evidence, build_followup_payload_options
from retrieval import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, COLLECTION_METADATA, PROCESSED_DIR, load_vector_db
from env_loader import resolve_project_path
from core import get_logger
from core.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = get_logger(__name__)


def require_app_client():
    return create_deepseek_client()


def ensure_build_runtime() -> None:
    ensure_embedding_key()


def safe_load_vector_db():
    try:
        vector_db = load_vector_db()
        return vector_db, int(vector_db._collection.count())
    except Exception:
        return None, 0


def safe_load_processed_manifest() -> dict[str, Any]:
    try:
        return load_manifest()
    except Exception:
        return {}


def safe_list_processed_corpora() -> list[dict]:
    try:
        return list_corpora()
    except Exception:
        return []


def copy_local_files_to_corpus(file_paths: list[str | Path], *, corpus_name: str) -> list[Path]:
    corpus_dir = ensure_corpus_dir(corpus_name, DOC_DIR)
    saved_paths: list[Path] = []
    for source in file_paths:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            continue
        file_path = corpus_dir / source_path.name
        shutil.copy2(source_path, file_path)
        saved_paths.append(file_path)
    return saved_paths


def rebuild_knowledge_base_from_local_files(
    file_paths: list[str | Path],
    *,
    corpus_name: str,
    model: str | None = None,
) -> tuple[list[Path], int]:
    ensure_build_runtime()
    saved_paths = copy_local_files_to_corpus(file_paths, corpus_name=corpus_name)
    if not saved_paths:
        raise ValueError("No valid local txt files were selected for import.")
    build_corpus_vector_db(
        corpus_name,
        file_paths=get_corpus_txt_files(corpus_name, DOC_DIR),
        llm_model=normalize_deepseek_model(model) if model else None,
    )
    file_count = len(get_all_txt_files(DOC_DIR))
    return saved_paths, file_count


def rebuild_selected_corpus(corpus_name: str, *, model: str | None = None) -> int:
    ensure_build_runtime()
    rebuild_existing_corpus(corpus_name, llm_model=normalize_deepseek_model(model) if model else None)
    return len(get_all_txt_files(DOC_DIR))


def update_corpus_incrementally(
    file_paths: list[str | Path],
    *,
    corpus_name: str,
    model: str | None = None,
) -> dict[str, Any]:
    ensure_build_runtime()
    saved_paths = copy_local_files_to_corpus(file_paths, corpus_name=corpus_name)
    if not saved_paths:
        raise ValueError("No valid local txt files were selected for incremental update.")
    _, _, stats = update_corpus_vector_db_incremental(
        corpus_name,
        file_paths=saved_paths,
        llm_model=normalize_deepseek_model(model) if model else None,
    )
    stats["saved_files"] = len(saved_paths)
    stats["file_count"] = len(get_all_txt_files(DOC_DIR))
    return stats


def preview_incremental_update(file_paths: list[str | Path], *, corpus_name: str) -> dict[str, Any]:
    from knowledge_base.service import _quick_load_text
    from preprocessing import load_corpus_result

    doc_dir = ensure_doc_dir(DOC_DIR)
    existing = load_corpus_result(corpus_name, PROCESSED_DIR)
    existing_by_name = {
        Path(str(document.relative_path).replace("\\", "/")).name: document
        for document in (existing.documents if existing else [])
    }
    added = 0
    updated = 0
    skipped = 0
    estimated_chunks = 0
    for source in file_paths:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            continue
        loaded = _quick_load_text(source_path, source_path.parent)
        existing_doc = existing_by_name.get(source_path.name)
        if existing_doc is None:
            added += 1
            estimated_chunks += _estimate_chunk_count(int(loaded.get("char_count", "0") or 0))
        elif existing_doc.normalized_sha1 == loaded["normalized_sha1"]:
            skipped += 1
        else:
            updated += 1
            estimated_chunks += _estimate_chunk_count(int(loaded.get("char_count", "0") or 0))
    return {
        "added_files": added,
        "updated_files": updated,
        "skipped_files": skipped,
        "estimated_chunks": estimated_chunks,
        "will_call_deepseek": added + updated > 0,
        "will_call_embedding": added + updated > 0,
    }


def _estimate_chunk_count(char_count: int) -> int:
    chunk_size = CHUNK_SIZE
    overlap = CHUNK_OVERLAP
    step = max(chunk_size - overlap, 1)
    if char_count <= 0:
        return 0
    return max(1, (char_count + step - 1) // step)


def load_alias_entries(corpus_name: str | None = None) -> dict[str, list[dict]] | list[dict]:
    payload = load_aliases(PROCESSED_DIR)
    if corpus_name is None:
        return payload
    return list(payload.get(corpus_name, []))


def save_alias_entries(corpus_name: str, entries: list[dict]) -> list[dict]:
    cleaned = []
    seen = set()
    for item in entries:
        alias = str(item.get("alias", "")).strip()
        canonical = str(item.get("canonical", "")).strip()
        note = str(item.get("note", "")).strip()
        if not alias or not canonical:
            continue
        key = (alias, canonical)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"alias": alias, "canonical": canonical, "note": note})
    payload = load_aliases(PROCESSED_DIR)
    payload[corpus_name] = cleaned
    save_aliases(payload, PROCESSED_DIR)
    return cleaned


def available_corpus_names() -> list[str]:
    names = set(list_available_corpora())
    for item in safe_list_processed_corpora():
        corpus_name = str(item.get("corpus_name", "")).strip()
        if corpus_name:
            names.add(corpus_name)
    return sorted(names)


def load_search_scope_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for corpus in safe_list_processed_corpora():
        corpus_name = str(corpus.get("corpus_name", "")).strip()
        if not corpus_name:
            continue
        storage_name = str(corpus.get("storage_name") or sanitize_corpus_name(corpus_name))
        volumes = _load_corpus_volumes(corpus_name, storage_name)
        catalog[corpus_name] = {
            "corpus_name": corpus_name,
            "storage_name": storage_name,
            "volumes": volumes,
            "has_unlabeled_chunks": any(volume.get("volume_index") is None for volume in volumes),
        }
    return catalog


def _load_corpus_volumes(corpus_name: str, storage_name: str | None = None) -> list[dict[str, Any]]:
    safe_name = storage_name or sanitize_corpus_name(corpus_name)
    chunks_path = PROCESSED_DIR / CORPORA_DIRNAME / safe_name / "chunks.jsonl"
    volumes: dict[int, dict[str, Any]] = {}
    unlabeled_count = 0
    if not chunks_path.exists():
        return []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            raw_index = row.get("volume_index", metadata.get("volume_index"))
            label = str(row.get("volume_label") or metadata.get("volume_label") or "").strip()
            try:
                volume_index = int(raw_index)
            except Exception:
                unlabeled_count += 1
                continue
            record = volumes.setdefault(
                volume_index,
                {"volume_index": volume_index, "volume_label": label or f"第{volume_index}卷", "chunk_count": 0},
            )
            record["chunk_count"] = int(record.get("chunk_count", 0)) + 1
            if label and not record.get("volume_label"):
                record["volume_label"] = label
    result = sorted(volumes.values(), key=lambda item: int(item.get("volume_index", 0)))
    if unlabeled_count:
        result.append({"volume_index": None, "volume_label": "未标卷内容", "chunk_count": unlabeled_count})
    return result


def load_workspace_snapshot() -> dict[str, Any]:
    manifest = safe_load_processed_manifest()
    corpora = safe_list_processed_corpora()
    indexed_files = get_all_txt_files(DOC_DIR)
    _, vector_chunk_count = safe_load_vector_db()
    return {
        "manifest": manifest,
        "corpora": corpora,
        "indexed_file_count": len(indexed_files),
        "vector_chunk_count": vector_chunk_count,
        "doc_dir": str(ensure_doc_dir(DOC_DIR)),
        "scope_catalog": load_search_scope_catalog(),
    }


def load_runtime_settings() -> dict[str, str]:
    return {
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", ""),
        "DEEPSEEK_API_BASE": os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        "DEEPSEEK_MODEL": os.getenv("DEEPSEEK_MODEL", "dsv4pro"),
        "SILICONFLOW_API_KEY": os.getenv("SILICONFLOW_API_KEY", ""),
        "SILICONFLOW_API_BASE": os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1"),
        "RAG_EMBEDDING_MODEL": os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
    }


def save_runtime_settings(settings: dict[str, str]) -> None:
    allowed = {
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_MODEL",
        "SILICONFLOW_API_KEY",
        "SILICONFLOW_API_BASE",
        "RAG_EMBEDDING_MODEL",
    }
    env_path = resolve_project_path(None, ".env")
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    incoming = {key: str(value).strip() for key, value in settings.items() if key in allowed}
    written: set[str] = set()
    output: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in incoming:
            output.append(f"{key}={incoming[key]}")
            os.environ[key] = incoming[key]
            written.add(key)
        else:
            output.append(line)
    for key, value in incoming.items():
        if key not in written:
            output.append(f"{key}={value}")
            os.environ[key] = value
    env_path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    _apply_runtime_settings(incoming)


def _apply_runtime_settings(settings: dict[str, str]) -> None:
    embedding_model = settings.get("RAG_EMBEDDING_MODEL")
    embedding_base = settings.get("SILICONFLOW_API_BASE")
    embedding_key = settings.get("SILICONFLOW_API_KEY")
    if embedding_model:
        retrieval_config.EMBEDDING_MODEL = embedding_model
        retrieval_config.COLLECTION_METADATA["embedding_model"] = embedding_model
        retrieval_embedding.EMBEDDING_MODEL = embedding_model
    if embedding_base:
        retrieval_config.EMBEDDING_API_BASE = embedding_base
        retrieval_embedding.EMBEDDING_API_BASE = embedding_base
    if embedding_key:
        retrieval_config.EMBEDDING_API_KEY = embedding_key
        retrieval_embedding.EMBEDDING_API_KEY = embedding_key


PACKAGE_MANIFEST_NAME = "storyrag_package_manifest.json"
PACKAGE_TYPE = "storyrag_kb_package"
PACKAGE_VERSION = 1


def export_knowledge_package(target_zip_path: str | Path, corpus_names: list[str] | None = None) -> dict[str, Any]:
    target_path = Path(target_zip_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = safe_load_processed_manifest()
    corpora = _selected_corpora_for_package(corpus_names)
    if not corpora:
        raise ValueError("No processed corpora are available to export.")
    if not Path(CHROMA_DB_DIR).exists():
        raise ValueError("Chroma DB directory does not exist. Build the knowledge base first.")

    package_manifest = {
        "package_type": PACKAGE_TYPE,
        "package_version": PACKAGE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "app_name": "StoryRAG",
        "corpora": corpora,
        "embedding_model": str(COLLECTION_METADATA.get("embedding_model", "")),
        "chroma_collection": CHROMA_COLLECTION_NAME,
        "source_paths": {
            "docs": str(DOC_DIR),
            "processed": str(PROCESSED_DIR),
            "chroma_db": str(CHROMA_DB_DIR),
        },
        "stats": {
            "document_count": sum(int(item.get("document_count", 0) or 0) for item in corpora),
            "chunk_count": sum(int(item.get("chunk_count", 0) or 0) for item in corpora),
            "relation_count": sum(int(item.get("relation_count", 0) or 0) for item in corpora),
            "global_chunk_count": int(manifest.get("chunk_count", 0) or 0),
        },
    }

    with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PACKAGE_MANIFEST_NAME, json.dumps(package_manifest, ensure_ascii=False, indent=2))
        _write_selected_docs_to_zip(archive, corpora)
        _write_selected_processed_to_zip(archive, corpora)
        aliases_path = PROCESSED_DIR / "aliases.json"
        if aliases_path.exists():
            archive.write(aliases_path, "processed/aliases.json")
        _write_directory_to_zip(archive, Path(CHROMA_DB_DIR), "chroma_db")

    return {
        "path": str(target_path),
        "corpus_count": len(corpora),
        "document_count": package_manifest["stats"]["document_count"],
        "chunk_count": package_manifest["stats"]["chunk_count"],
    }


def inspect_knowledge_package(zip_path: str | Path) -> dict[str, Any]:
    path = Path(zip_path).resolve()
    payload = _read_package_manifest(path)
    corpora = payload.get("corpora", []) if isinstance(payload.get("corpora"), list) else []
    local_names = set(available_corpus_names())
    names = [str(item.get("corpus_name", "")).strip() for item in corpora if isinstance(item, dict)]
    conflicts = sorted(name for name in names if name and name in local_names)
    return {
        "path": str(path),
        "manifest": payload,
        "corpora": corpora,
        "corpus_names": names,
        "conflicts": conflicts,
        "embedding_model": str(payload.get("embedding_model", "")),
        "local_embedding_model": str(COLLECTION_METADATA.get("embedding_model", "")),
    }


def import_knowledge_package(zip_path: str | Path, *, overwrite_corpora: bool = False) -> dict[str, Any]:
    info = inspect_knowledge_package(zip_path)
    package_model = str(info.get("embedding_model") or "").strip()
    local_model = str(info.get("local_embedding_model") or "").strip()
    if package_model and local_model and package_model != local_model:
        raise ValueError(
            f"Embedding model mismatch: package uses '{package_model}', current app uses '{local_model}'. "
            "Please use the same embedding model or rebuild the imported corpus."
        )
    conflicts = list(info.get("conflicts", []))
    if conflicts and not overwrite_corpora:
        raise ValueError(f"Package conflicts with existing corpora: {', '.join(conflicts)}")

    zip_path = Path(zip_path).resolve()
    with tempfile.TemporaryDirectory(prefix="storyrag_import_") as tmp:
        temp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_dir)

        corpora = [item for item in info.get("corpora", []) if isinstance(item, dict)]
        corpus_names = [str(item.get("corpus_name", "")).strip() for item in corpora if str(item.get("corpus_name", "")).strip()]
        if not corpus_names:
            raise ValueError("Package contains no valid corpora.")

        local_db = load_vector_db()
        for corpus_name in corpus_names:
            _delete_local_corpus(local_db, corpus_name)

        _copy_imported_docs(temp_dir, corpus_names)
        _copy_imported_processed(temp_dir, corpora)
        _merge_imported_aliases(temp_dir, corpus_names)
        _import_chroma_corpora(temp_dir / "chroma_db", local_db, corpus_names)
        _rewrite_global_processed_manifest()

    return {"corpus_names": corpus_names, "corpus_count": len(corpus_names), "conflicts": conflicts}


def _selected_corpora_for_package(corpus_names: list[str] | None) -> list[dict[str, Any]]:
    selected = {name.strip() for name in (corpus_names or []) if str(name).strip()}
    corpora = []
    for item in safe_list_processed_corpora():
        corpus_name = str(item.get("corpus_name", "")).strip()
        if not corpus_name:
            continue
        if selected and corpus_name not in selected:
            continue
        corpora.append(dict(item))
    return corpora


def _write_selected_docs_to_zip(archive: zipfile.ZipFile, corpora: list[dict[str, Any]]) -> None:
    names = {str(item.get("corpus_name", "")).strip() for item in corpora if str(item.get("corpus_name", "")).strip()}
    for corpus_name in names:
        corpus_dir = DOC_DIR / corpus_name
        if corpus_dir.exists():
            _write_directory_to_zip(archive, corpus_dir, f"docs/{corpus_name}")
    for path in get_all_txt_files(DOC_DIR):
        try:
            relative = path.relative_to(DOC_DIR)
        except Exception:
            continue
        if relative.parts and relative.parts[0] in names:
            continue
        # Legacy root-level docs are only exported if their processed manifest names them.
        for corpus in corpora:
            for doc in corpus.get("documents", []) or []:
                if isinstance(doc, dict) and doc.get("relative_path") == str(relative).replace("\\", "/"):
                    archive.write(path, f"docs/{relative.as_posix()}")


def _write_selected_processed_to_zip(archive: zipfile.ZipFile, corpora: list[dict[str, Any]]) -> None:
    manifest_path = PROCESSED_DIR / "manifest.json"
    if manifest_path.exists():
        archive.write(manifest_path, "processed/manifest.json")
    for corpus in corpora:
        storage_name = str(corpus.get("storage_name") or sanitize_corpus_name(str(corpus.get("corpus_name", ""))))
        corpus_dir = PROCESSED_DIR / CORPORA_DIRNAME / storage_name
        if corpus_dir.exists():
            _write_directory_to_zip(archive, corpus_dir, f"processed/{CORPORA_DIRNAME}/{storage_name}")


def _write_directory_to_zip(archive: zipfile.ZipFile, source_dir: Path, prefix: str) -> None:
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return
    for path in source_dir.rglob("*"):
        if path.is_file():
            archive.write(path, f"{prefix}/{path.relative_to(source_dir).as_posix()}")


def _read_package_manifest(zip_path: Path) -> dict[str, Any]:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(PACKAGE_MANIFEST_NAME) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
    except KeyError as exc:
        raise ValueError("This is not a StoryRAG knowledge package: missing package manifest.") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid zip file.") from exc
    if not isinstance(payload, dict) or payload.get("package_type") != PACKAGE_TYPE:
        raise ValueError("This is not a StoryRAG knowledge package.")
    return payload


def _delete_local_corpus(vector_db, corpus_name: str) -> None:
    try:
        existing = vector_db._collection.get(where={"corpus_name": corpus_name}, include=[])
        ids = existing.get("ids", []) if isinstance(existing, dict) else []
        if ids:
            vector_db.delete(ids=ids)
    except Exception:
        logger.warning("删除语料 %r 的向量记录失败,继续清理本地文件", corpus_name, exc_info=True)
    corpus_dir = DOC_DIR / corpus_name
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    processed_dir = corpus_processed_dir(corpus_name, PROCESSED_DIR)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)


def _copy_imported_docs(temp_dir: Path, corpus_names: list[str]) -> None:
    imported_docs = temp_dir / "docs"
    if not imported_docs.exists():
        return
    ensure_doc_dir(DOC_DIR)
    for source in imported_docs.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(imported_docs)
        target = DOC_DIR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_imported_processed(temp_dir: Path, corpora: list[dict[str, Any]]) -> None:
    imported_root = temp_dir / "processed" / CORPORA_DIRNAME
    if not imported_root.exists():
        return
    target_root = PROCESSED_DIR / CORPORA_DIRNAME
    target_root.mkdir(parents=True, exist_ok=True)
    for corpus in corpora:
        corpus_name = str(corpus.get("corpus_name", "")).strip()
        storage_name = str(corpus.get("storage_name") or sanitize_corpus_name(corpus_name))
        source = imported_root / storage_name
        target = target_root / storage_name
        if source.exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)


def _merge_imported_aliases(temp_dir: Path, corpus_names: list[str]) -> None:
    aliases_path = temp_dir / "processed" / "aliases.json"
    if not aliases_path.exists():
        return
    try:
        imported = json.loads(aliases_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(imported, dict):
        return
    current = load_aliases(PROCESSED_DIR)
    for corpus_name in corpus_names:
        if corpus_name in imported:
            current[corpus_name] = imported.get(corpus_name, [])
    save_aliases(current, PROCESSED_DIR)


def _import_chroma_corpora(imported_chroma_dir: Path, local_db, corpus_names: list[str]) -> None:
    if not imported_chroma_dir.exists():
        raise ValueError("Package does not contain chroma_db.")
    imported_db = load_vector_db(str(imported_chroma_dir))
    for corpus_name in corpus_names:
        payload = imported_db._collection.get(
            where={"corpus_name": corpus_name},
            include=["documents", "metadatas", "embeddings"],
        )
        ids = payload.get("ids", []) if isinstance(payload, dict) else []
        documents = payload.get("documents", []) if isinstance(payload, dict) else []
        metadatas = payload.get("metadatas", []) if isinstance(payload, dict) else []
        embeddings = payload.get("embeddings", None) if isinstance(payload, dict) else None
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        if not ids:
            continue
        local_db._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def _rewrite_global_processed_manifest() -> None:
    from preprocessing.storage import _write_global_index

    _write_global_index(PROCESSED_DIR)


def ask_story_question(
    *,
    question: str,
    corpus_names: list[str] | None = None,
    search_scope: dict[str, Any] | None = None,
    question_history: list[dict[str, str] | str] | None = None,
    selected_contexts: list[dict] | None = None,
    model: str | None = None,
    client=None,
    vector_db=None,
) -> dict[str, Any]:
    active_model = normalize_deepseek_model(model) if model else None
    active_client = client or require_app_client()
    active_vector_db = vector_db
    if active_vector_db is None:
        active_vector_db, _ = safe_load_vector_db()
    if active_vector_db is None:
        raise RuntimeError("Vector DB is not available. Build or load a corpus first.")

    retrieval_result, validated_res = answer_with_evidence(
        client=active_client,
        vector_db=active_vector_db,
        question=question,
        corpus_names=corpus_names,
        search_scope=search_scope,
        question_history=question_history,
        selected_contexts=selected_contexts,
        model=active_model,
    )
    return {
        "retrieval_result": retrieval_result,
        "validated_response": validated_res,
        "followup_context_options": build_followup_payload_options(retrieval_result),
    }
