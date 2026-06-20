from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from env_loader import load_project_env
from core.config import PROCESSED_DIR

from .schema import PreprocessingResult
from .schema import ChapterArtifact, ChunkArtifact, RelationArtifact, SourceDocumentArtifact


load_project_env()

CORPORA_DIRNAME = "corpora"
ALIASES_FILENAME = "aliases.json"


def ensure_processed_dir(directory: Path = PROCESSED_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def corpus_processed_dir(corpus_name: str, output_dir: Path = PROCESSED_DIR) -> Path:
    safe_name = sanitize_corpus_name(corpus_name)
    return ensure_processed_dir(output_dir) / CORPORA_DIRNAME / safe_name


def persist_preprocessing_result(
    result: PreprocessingResult,
    output_dir: Path = PROCESSED_DIR,
) -> Path:
    output_dir = ensure_processed_dir(output_dir)
    corpora_root = output_dir / CORPORA_DIRNAME
    corpora_root.mkdir(parents=True, exist_ok=True)

    if result.documents:
        by_corpus = _split_result_by_corpus(result)
        for corpus_name, corpus_result in by_corpus.items():
            corpus_dir = corpus_processed_dir(corpus_name, output_dir)
            _persist_single_result(corpus_result, corpus_dir)

    _write_global_index(output_dir)
    return output_dir


def load_manifest(output_dir: Path = PROCESSED_DIR, corpus_name: str | None = None) -> dict:
    if corpus_name:
        manifest_path = corpus_processed_dir(corpus_name, output_dir) / "manifest.json"
    else:
        manifest_path = ensure_processed_dir(output_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_corpus_result(corpus_name: str, output_dir: Path = PROCESSED_DIR) -> PreprocessingResult | None:
    corpus_dir = corpus_processed_dir(corpus_name, output_dir)
    if not corpus_dir.exists():
        return None
    manifest = load_manifest(output_dir, corpus_name)
    return PreprocessingResult(
        pipeline_version=str(manifest.get("pipeline_version", "unknown")),
        generated_at=str(manifest.get("generated_at", "")),
        documents=[SourceDocumentArtifact(**row) for row in _read_jsonl(corpus_dir / "documents.jsonl")],
        chapters=[ChapterArtifact(**row) for row in _read_jsonl(corpus_dir / "chapters.jsonl")],
        chunks=[ChunkArtifact(**row) for row in _read_jsonl(corpus_dir / "chunks.jsonl")],
        relations=[RelationArtifact(**row) for row in _read_jsonl(corpus_dir / "relations.jsonl")],
    )


def persist_corpus_result(
    corpus_name: str,
    result: PreprocessingResult,
    output_dir: Path = PROCESSED_DIR,
) -> Path:
    corpus_dir = corpus_processed_dir(corpus_name, output_dir)
    _persist_single_result(result, corpus_dir)
    _write_global_index(ensure_processed_dir(output_dir))
    return corpus_dir


def list_corpora(output_dir: Path = PROCESSED_DIR) -> list[dict]:
    manifest = load_manifest(output_dir)
    corpora = manifest.get("corpora", [])
    if isinstance(corpora, list):
        return corpora
    return []


def sanitize_corpus_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(name).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "default"


def load_aliases(output_dir: Path = PROCESSED_DIR) -> dict[str, list[dict]]:
    aliases_path = ensure_processed_dir(output_dir) / ALIASES_FILENAME
    if not aliases_path.exists():
        return {}
    payload = json.loads(aliases_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def save_aliases(payload: dict[str, list[dict]], output_dir: Path = PROCESSED_DIR) -> Path:
    aliases_path = ensure_processed_dir(output_dir) / ALIASES_FILENAME
    _write_json(aliases_path, payload)
    return aliases_path


def _persist_single_result(result: PreprocessingResult, corpus_dir: Path) -> None:
    temp_dir = corpus_dir.with_name(f"{corpus_dir.name}_tmp")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    _write_json(temp_dir / "manifest.json", result.build_manifest())
    _write_jsonl(temp_dir / "documents.jsonl", [item.to_dict() for item in result.documents])
    _write_jsonl(temp_dir / "chapters.jsonl", [item.to_dict() for item in result.chapters])
    _write_jsonl(temp_dir / "chunks.jsonl", [item.to_dict() for item in result.chunks])
    _write_jsonl(temp_dir / "relations.jsonl", [item.to_dict() for item in result.relations])

    backup_dir = corpus_dir.with_name(f"{corpus_dir.name}_bak")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    if corpus_dir.exists():
        corpus_dir.rename(backup_dir)

    try:
        temp_dir.rename(corpus_dir)
    except Exception:
        if backup_dir.exists() and not corpus_dir.exists():
            backup_dir.rename(corpus_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def _write_global_index(output_dir: Path) -> None:
    corpora_root = ensure_processed_dir(output_dir) / CORPORA_DIRNAME
    corpora: list[dict] = []
    total_documents = 0
    total_chapters = 0
    total_chunks = 0
    total_relations = 0
    pipeline_version = "unknown"
    generated_at = ""

    if corpora_root.exists():
        for corpus_dir in sorted(path for path in corpora_root.iterdir() if path.is_dir()):
            manifest_path = corpus_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            corpus_record = {
                "corpus_name": _read_corpus_name(payload, corpus_dir.name),
                "storage_name": corpus_dir.name,
                "document_count": int(payload.get("document_count", 0) or 0),
                "chapter_count": int(payload.get("chapter_count", 0) or 0),
                "chunk_count": int(payload.get("chunk_count", 0) or 0),
                "relation_count": int(payload.get("relation_count", 0) or 0),
                "documents": payload.get("documents", []),
            }
            corpora.append(corpus_record)
            total_documents += corpus_record["document_count"]
            total_chapters += corpus_record["chapter_count"]
            total_chunks += corpus_record["chunk_count"]
            total_relations += corpus_record["relation_count"]
            pipeline_version = payload.get("pipeline_version", pipeline_version)
            generated_at = max(generated_at, str(payload.get("generated_at", "")))

    global_manifest = {
        "pipeline_version": pipeline_version,
        "generated_at": generated_at,
        "document_count": total_documents,
        "chapter_count": total_chapters,
        "chunk_count": total_chunks,
        "relation_count": total_relations,
        "corpora": corpora,
    }
    _write_json(output_dir / "manifest.json", global_manifest)


def _split_result_by_corpus(result: PreprocessingResult) -> dict[str, PreprocessingResult]:
    documents_by_id = {item.doc_id: item for item in result.documents}
    chapters_by_doc: dict[str, list] = defaultdict(list)
    chunks_by_doc: dict[str, list] = defaultdict(list)
    relations_by_doc: dict[str, list] = defaultdict(list)

    for chapter in result.chapters:
        chapters_by_doc[chapter.doc_id].append(chapter)
    for chunk in result.chunks:
        chunks_by_doc[chunk.doc_id].append(chunk)
    for relation in result.relations:
        relations_by_doc[relation.doc_id].append(relation)

    grouped: dict[str, PreprocessingResult] = {}
    docs_by_corpus: dict[str, list] = defaultdict(list)
    for document in result.documents:
        docs_by_corpus[document.corpus_name].append(document)

    for corpus_name, documents in docs_by_corpus.items():
        doc_ids = {item.doc_id for item in documents}
        grouped[corpus_name] = PreprocessingResult(
            pipeline_version=result.pipeline_version,
            generated_at=result.generated_at,
            documents=documents,
            chapters=[chapter for doc_id in doc_ids for chapter in chapters_by_doc.get(doc_id, [])],
            chunks=[chunk for doc_id in doc_ids for chunk in chunks_by_doc.get(doc_id, [])],
            relations=[relation for doc_id in doc_ids for relation in relations_by_doc.get(doc_id, [])],
        )
    return grouped


def _read_corpus_name(manifest: dict, default_name: str) -> str:
    documents = manifest.get("documents", [])
    if isinstance(documents, list):
        for item in documents:
            if isinstance(item, dict) and item.get("corpus_name"):
                return str(item["corpus_name"])
    return default_name


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
