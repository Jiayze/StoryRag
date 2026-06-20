from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceDocumentArtifact:
    doc_id: str
    source_path: str
    relative_path: str
    doc_name: str
    corpus_name: str
    encoding: str
    raw_sha1: str
    normalized_sha1: str
    char_count: int
    chapter_count: int
    chunk_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChapterArtifact:
    chapter_id: str
    doc_id: str
    source_path: str
    doc_name: str
    corpus_name: str
    title: str
    chapter_index: int
    char_start: int
    char_end: int
    text: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkArtifact:
    chunk_id: str
    doc_id: str
    chapter_id: str
    source_path: str
    doc_name: str
    corpus_name: str
    chapter_title: str
    chapter_index: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RelationArtifact:
    relation_id: str
    doc_id: str
    chapter_id: str
    source_path: str
    doc_name: str
    corpus_name: str
    person_a: str
    person_b: str
    relation_type: str
    evidence_chunk_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreprocessingResult:
    pipeline_version: str
    generated_at: str
    documents: list[SourceDocumentArtifact]
    chapters: list[ChapterArtifact]
    chunks: list[ChunkArtifact]
    relations: list[RelationArtifact]

    def build_manifest(self) -> dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "generated_at": self.generated_at,
            "document_count": len(self.documents),
            "chapter_count": len(self.chapters),
            "chunk_count": len(self.chunks),
            "relation_count": len(self.relations),
            "documents": [
                {
                    "doc_id": item.doc_id,
                    "doc_name": item.doc_name,
                    "corpus_name": item.corpus_name,
                    "relative_path": item.relative_path,
                    "raw_sha1": item.raw_sha1,
                    "normalized_sha1": item.normalized_sha1,
                    "chapter_count": item.chapter_count,
                    "chunk_count": item.chunk_count,
                }
                for item in self.documents
            ],
        }
