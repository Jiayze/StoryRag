from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.config import (
    CHAPTER_SUMMARY_LENGTH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    KEYWORD_LIMIT,
    MIN_PERSON_FREQUENCY,
    MIN_RELATION_PERSON_FREQUENCY,
    PREPROCESS_CONCURRENCY,
)
from core import get_logger
from .enrichment import build_enricher, merge_keywords, merge_metadata
from .schema import (
    ChapterArtifact,
    ChunkArtifact,
    PreprocessingResult,
    RelationArtifact,
    SourceDocumentArtifact,
)


logger = get_logger(__name__)


PIPELINE_VERSION = "heavy-preprocess-v7-ds"
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
DEFAULT_CHAPTER_TITLE = "Main Body"

CHAPTER_PATTERNS = (
    re.compile(r"^\s*第\s*[0-9零一二三四五六七八九十百千两]+\s*[章节卷部篇回幕集]\s*.*$"),
    re.compile(r"^\s*chapter\s+\d+.*$", re.IGNORECASE),
    re.compile(r"^\s*(prologue|epilogue)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(序章|楔子|尾声|后记)\s*$"),
)

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{1,31}")
CN_NAME_PATTERN = re.compile(r"(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{2,4}(?![\u4e00-\u9fff])")
EN_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?\b")
DOTTED_NAME_PATTERN = re.compile(
    r"(?<![\u4e00-\u9fffA-Za-z])"
    r"(?:[\u4e00-\u9fff]{1,4}|[A-Za-z]{1,12})·(?:[\u4e00-\u9fff]{1,4}|[A-Za-z]{1,12})"
    r"(?![\u4e00-\u9fffA-Za-z])"
)
TITLED_NAME_PATTERN = re.compile(
    r"([\u4e00-\u9fff]{2,4}|[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?|[\u4e00-\u9fffA-Za-z]{1,4}·[\u4e00-\u9fffA-Za-z]{1,4})"
    r"(先生|太太|夫人|教授|小姐|同学|女士)"
)
INTRO_NAME_PATTERN = re.compile(
    r"(?:名叫|叫做|叫|名为)([\u4e00-\u9fff]{2,4}|[A-Z][a-z]{1,20}|[\u4e00-\u9fffA-Za-z]{1,4}·[\u4e00-\u9fffA-Za-z]{1,4})"
)
NAME_CONTEXT_PATTERN = re.compile(r"(先生|太太|小姐|教授|同学|女士|男孩|女孩|夫人|老师|名叫|叫做|叫)")
LINE_NOISE_PATTERN = re.compile(r"^\s*(page\s*\d+|\d+)\s*$", re.IGNORECASE)
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LOCATION_HINT_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{1,20}(?:路|街|巷|村|镇|城|国|省|市|县|山|湖|岛|湾|宫|堡|苑|庄园|学校|学院)")
OBJECT_HINT_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{1,20}(?:剑|石|镜|戒指|信|地图|魔杖|箱子|钥匙|书|日记|项链)")
EVENT_HINT_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{1,20}(?:遇见|收到|发现|进入|离开|前往|逃离|战斗|袭击|死亡|出生|出现)")
ALIAS_CONNECTOR_PATTERN = re.compile(r"(又叫|也叫|称为|被称为|化名|别名)")
RELATION_PATTERNS = (
    ("family", re.compile(r"(父亲|母亲|儿子|女儿|哥哥|姐姐|弟弟|妹妹|叔叔|阿姨|姨妈|舅舅|姑妈|表哥|表姐|表弟|表妹|夫妇|夫妻)")),
    ("friend", re.compile(r"(朋友|同伴|伙伴|搭档|挚友)")),
    ("enemy", re.compile(r"(敌人|仇人|死对头|对手)")),
    ("mentor", re.compile(r"(老师|教授|导师|师父)")),
    ("helper", re.compile(r"(帮助|照顾|保护|引导)")),
)

VOLUME_PATTERN = re.compile(r"第\s*([0-9零〇一二两三四五六七八九十百]+)\s*([卷册部])")
PLACEHOLDER_TITLE_PATTERNS = (
    re.compile(r"^\s*第[0-9零〇一二两三四五六七八九十百]+\s*[卷册部]\s*(插图|彩页)\s*$"),
    re.compile(r"^\s*(卷首插图|卷末插图|插图|彩页|封面|扉页|人物设定图)\s*$"),
)
PLACEHOLDER_TEXT_PATTERNS = (
    re.compile(r"(卷首插图|卷末插图|人物设定图|彩页|封面|扉页)"),
)
CN_NUMERAL_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

# 分段与抽取参数已收口至 core.config(见上方导入)

STOPWORDS = {
    "他们",
    "她们",
    "我们",
    "你们",
    "自己",
    "这个",
    "那个",
    "这里",
    "那里",
    "因为",
    "所以",
    "如果",
    "但是",
    "然后",
    "先生",
    "太太",
    "小姐",
    "时候",
    "事情",
    "地方",
    "里面",
    "外面",
    "已经",
    "没有",
    "不会",
    "不是",
    "就是",
    "什么",
    "拜托",
    "接着",
    "不错",
    "没错",
    "最后",
    "好吧",
    "不准",
    "不时",
    "不用",
    "不行",
    "不了",
    "绝不",
    "三天",
    "孩子",
    "面前",
    "侮辱",
    "教室",
    "妈妈",
    "爸爸",
    "男孩",
    "女孩",
    "女人",
    "男人",
    "老人",
    "人们",
    "大家",
    "之后",
    "不过",
    "可街",
    "在街",
    "一切",
    "一样",
    "后来",
    "这是",
    "的是",
    "这跟",
    "一边",
    "走开",
    "还有",
    "是的",
    "突然",
    "老头",
    "小子",
    "要我",
    "上坐",
    "下的",
    "话筒",
    "哎呀",
    "老爸",
    "主人",
    "和第",
}

NON_PERSON_TERMS = {
    "是他",
    "是啊",
    "是吧",
    "好了",
    "很好",
    "当然",
    "我想",
    "走吧",
    "那么",
    "这时",
    "这么",
    "而且",
    "谢谢",
    "多谢",
    "啊哈",
    "巨怪",
    "级长",
    "好吗",
    "顿时",
    "低声",
    "确实",
    "立刻",
    "光轮",
    "分之",
    "终于",
    "记住",
    "哈利想",
    "海格又",
    "然而",
    "住嘴",
    "站台",
    "天哪",
    "再见",
    "怎么",
    "快点",
    "闭嘴",
    "糟糕",
    "显然",
    "真的",
    "其实",
    "我要",
    "让我",
    "你好",
    "那好",
    "另外",
    "比如",
    "像我",
    "晚安",
    "爪子",
    "时所",
}

BANNED_ENTITY_FRAGMENTS = (
    "因为",
    "所以",
    "如果",
    "但是",
    "然后",
    "他们",
    "她们",
    "我们",
    "你们",
    "自己",
    "一个",
    "一种",
    "没有",
    "不会",
    "发现",
    "看到",
    "事情",
    "今天",
    "昨天",
    "明天",
    "早上",
    "晚上",
    "中午",
)

COMMON_SURNAME_PREFIXES = {
    "哈利",
    "海格",
    "邓布",
    "赫敏",
    "罗恩",
    "德思",
    "佩妮",
    "弗农",
    "达力",
    "麦格",
    "汤姆",
    "马尔",
    "斯内",
}

BAD_NAME_ENDINGS = ("说", "问", "看", "听", "走", "来", "去", "道", "把", "和", "跟", "在", "是")
GENERIC_ROLE_WORDS = {"麻瓜", "孩子", "男孩", "女孩", "妈妈", "爸爸", "女人", "男人", "老人", "人们", "大家"}
LOCATION_SPLIT_MARKERS = ("家住", "住在", "在", "到", "从", "往", "向", "进", "出", "只见", "当")
NOISY_NAME_PREFIXES = (
    "我叫",
    "我是",
    "叫阿",
    "叫",
    "长阿",
    "让",
    "我要",
    "让我",
    "交给",
    "伙伴",
    "像我",
    "比如",
)
NOISY_NAME_SUFFIXES = ("则成", "就是", "来了", "说道", "说", "问道", "问")
MENTOR_TITLE_MARKERS = ("教授", "老师", "导师")
MENTOR_ACTION_HINTS = ("上课", "教", "教学", "指导", "教导", "训练", "罚", "训", "批评", "告诉", "建议")


@dataclass(slots=True)
class LoadedText:
    path: Path
    relative_path: str
    encoding: str
    raw_text: str
    normalized_text: str
    raw_sha1: str
    normalized_sha1: str


@dataclass(slots=True)
class EntityLexicon:
    persons: set[str]
    person_frequency: dict[str, int]
    strong_persons: set[str]
    titled_persons: set[str]


def preprocess_files(
    file_paths: list[Path],
    base_dir: Path | None = None,
    *,
    use_llm_enrichment: bool | None = None,
    llm_model: str | None = None,
) -> PreprocessingResult:
    if not file_paths:
        raise ValueError("No text files were provided for preprocessing.")

    base_dir = (base_dir or _common_base_dir(file_paths)).resolve()
    documents: list[SourceDocumentArtifact] = []
    chapters: list[ChapterArtifact] = []
    chunks: list[ChunkArtifact] = []
    enricher = build_enricher(enabled=use_llm_enrichment, model=llm_model)
    logger.info(
        "Preprocessing started: "
        f"documents={len(file_paths)}, llm_enrichment={'on' if enricher else 'off'}."
    )

    for doc_number, path in enumerate(file_paths, start=1):
        logger.info(f"Loading document {doc_number}/{len(file_paths)}: {path.name}")
        loaded = _load_text(path, base_dir=base_dir)
        lexicon = _build_entity_lexicon(loaded.normalized_text)
        doc_id = _stable_id("doc", loaded.relative_path, loaded.normalized_sha1)

        doc_chapters = _split_into_chapters(loaded)
        logger.info(
            f"Parsed document {path.name}: chars={len(loaded.normalized_text)}, chapters={len(doc_chapters)}."
        )
        doc_chunks = _chunk_document(
            loaded,
            doc_id=doc_id,
            chapters=doc_chapters,
            lexicon=lexicon,
            enricher=enricher,
        )
        role_index_chunks = _build_role_index_chunks(
            loaded,
            doc_id=doc_id,
            chapters=doc_chapters,
            chunks=doc_chunks,
            lexicon=lexicon,
            enricher=enricher,
            starting_chunk_index=len(doc_chunks),
        )
        if role_index_chunks:
            doc_chunks.extend(role_index_chunks)
            logger.info(f"Added {len(role_index_chunks)} synthetic role index chunks for {path.name}.")

        documents.append(
            SourceDocumentArtifact(
                doc_id=doc_id,
                source_path=str(loaded.path),
                relative_path=loaded.relative_path,
                doc_name=loaded.path.name,
                corpus_name=_infer_corpus_name(loaded.relative_path),
                encoding=loaded.encoding,
                raw_sha1=loaded.raw_sha1,
                normalized_sha1=loaded.normalized_sha1,
                char_count=len(loaded.normalized_text),
                chapter_count=len(doc_chapters),
                chunk_count=len(doc_chunks),
                metadata={
                    "base_dir": str(base_dir),
                    "person_lexicon_size": len(lexicon.persons),
                    "llm_enrichment": bool(enricher),
                    "pipeline_version": PIPELINE_VERSION,
                    "llm_model": getattr(enricher, "model", "") if enricher else "",
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                },
            )
        )
        chapters.extend(doc_chapters)
        chunks.extend(doc_chunks)
        logger.info(
            f"Document {path.name} complete: cumulative_chapters={len(chapters)}, cumulative_chunks={len(chunks)}."
        )

    relations = _build_relations(chunks)
    logger.info(
        f"Preprocessing completed: documents={len(documents)}, chapters={len(chapters)}, "
        f"chunks={len(chunks)}, relations={len(relations)}."
    )

    return PreprocessingResult(
        pipeline_version=PIPELINE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        documents=documents,
        chapters=chapters,
        chunks=chunks,
        relations=relations,
    )


def chunk_to_payload(chunk: ChunkArtifact) -> tuple[str, dict[str, object], str]:
    metadata = {
        "source": chunk.source_path,
        "doc_name": chunk.doc_name,
        "corpus_name": chunk.corpus_name,
        "doc_id": chunk.doc_id,
        "chapter_id": chunk.chapter_id,
        "chapter": chunk.chapter_title,
        "chapter_index": chunk.chapter_index,
        "chunk_index": chunk.chunk_index,
        "chunk_id": chunk.chunk_id,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "chapter_summary": chunk.summary,
        "keywords": chunk.keywords,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "persons": chunk.metadata.get("persons", []),
        "locations": chunk.metadata.get("locations", []),
        "events": chunk.metadata.get("events", []),
        "objects": chunk.metadata.get("objects", []),
        "aliases": chunk.metadata.get("aliases", []),
        "relations": chunk.metadata.get("relations", []),
        "volume_label": chunk.metadata.get("volume_label"),
        "volume_index": chunk.metadata.get("volume_index"),
        "is_placeholder_chunk": bool(chunk.metadata.get("is_placeholder_chunk")),
    }
    metadata.update(chunk.metadata)
    return chunk.chunk_id, metadata, chunk.text


def _build_entity_lexicon(text: str) -> EntityLexicon:
    frequency = Counter()
    context_hits = Counter()
    strong_candidates: set[str] = set()
    titled_candidates: set[str] = set()

    for sentence in re.split(r"[。！？!?]\s*", text):
        if not sentence:
            continue
        candidates = _raw_person_candidates(sentence)
        sentence_strong_candidates: set[str] = set()
        for candidate in _titled_person_candidates(sentence):
            normalized = _normalize_person_candidate(candidate)
            if normalized:
                titled_candidates.add(normalized)
        for candidate in _strong_person_candidates(sentence):
            normalized = _normalize_person_candidate(candidate)
            if normalized:
                strong_candidates.add(normalized)
                sentence_strong_candidates.add(normalized)
        for candidate in candidates:
            normalized = _normalize_person_candidate(candidate)
            if not normalized:
                continue
            frequency[normalized] += 1
            if _has_person_like_context(sentence, candidate, normalized, sentence_strong_candidates):
                context_hits[normalized] += 1

    persons = {
        candidate
        for candidate, count in frequency.items()
        if _should_keep_person_candidate(
            candidate,
            count=count,
            context_hit_count=context_hits[candidate],
            strong_candidates=strong_candidates,
        )
    }
    persons.update(strong_candidates)
    return EntityLexicon(
        persons=persons,
        person_frequency=dict(frequency),
        strong_persons=strong_candidates,
        titled_persons=titled_candidates,
    )


def _load_text(path: Path, base_dir: Path) -> LoadedText:
    raw_text, encoding = _read_text_with_fallback(path)
    normalized_text = _normalize_text(raw_text)
    raw_sha1 = _sha1_hex(raw_text)
    normalized_sha1 = _sha1_hex(normalized_text)
    relative_path = path.resolve().relative_to(base_dir.resolve()).as_posix()

    return LoadedText(
        path=path.resolve(),
        relative_path=relative_path,
        encoding=encoding,
        raw_text=raw_text,
        normalized_text=normalized_text,
        raw_sha1=raw_sha1,
        normalized_sha1=normalized_sha1,
    )


def _read_text_with_fallback(path: Path) -> tuple[str, str]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode {path} with {', '.join(ENCODINGS)}: {last_error}",
    )


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_PATTERN.sub("", text)

    normalized_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if LINE_NOISE_PATTERN.match(line):
            continue
        if line:
            line = re.sub(r"\s+", " ", line)
            normalized_lines.append(line)
            continue
        if normalized_lines and normalized_lines[-1] == "":
            continue
        normalized_lines.append("")

    cleaned = "\n".join(normalized_lines)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned.strip()


def _split_into_chapters(loaded: LoadedText) -> list[ChapterArtifact]:
    lines = loaded.normalized_text.splitlines(keepends=True)
    positions: list[tuple[int, str]] = []
    offset = 0

    for line in lines:
        stripped = line.strip()
        if stripped and _is_chapter_heading(stripped):
            positions.append((offset, stripped))
        offset += len(line)

    if not positions:
        return [
            _build_chapter_artifact(
                loaded,
                doc_id=None,
                chapter_index=0,
                title=DEFAULT_CHAPTER_TITLE,
                char_start=0,
                char_end=len(loaded.normalized_text),
                text=loaded.normalized_text,
            )
        ]

    chapters: list[ChapterArtifact] = []
    for index, (char_start, title) in enumerate(positions):
        char_end = positions[index + 1][0] if index + 1 < len(positions) else len(loaded.normalized_text)
        text = loaded.normalized_text[char_start:char_end].strip()
        if not text:
            continue
        chapters.append(
            _build_chapter_artifact(
                loaded,
                doc_id=None,
                chapter_index=index,
                title=title,
                char_start=char_start,
                char_end=char_end,
                text=text,
            )
        )

    return chapters or [
        _build_chapter_artifact(
            loaded,
            doc_id=None,
            chapter_index=0,
            title=DEFAULT_CHAPTER_TITLE,
            char_start=0,
            char_end=len(loaded.normalized_text),
            text=loaded.normalized_text,
        )
    ]


def _chunk_document(
    loaded: LoadedText,
    *,
    doc_id: str,
    chapters: list[ChapterArtifact],
    lexicon: EntityLexicon,
    enricher,
) -> list[ChunkArtifact]:
    chunks: list[ChunkArtifact] = []
    chunk_index = 0

    total_chapters = len(chapters)
    for chapter_number, chapter in enumerate(chapters, start=1):
        logger.info(
            f"Processing chapter {chapter_number}/{total_chapters}: {chapter.title} "
            f"(chars={len(chapter.text)})."
        )
        chapter.chapter_id = _stable_id(
            "chapter",
            doc_id,
            str(chapter.chapter_index),
            chapter.title,
            str(chapter.char_start),
            str(chapter.char_end),
        )
        chapter.doc_id = doc_id
        chapter.keywords = _extract_keywords(chapter.text, limit=KEYWORD_LIMIT)
        chapter.summary = _build_summary(chapter.text, limit=CHAPTER_SUMMARY_LENGTH)
        chapter.metadata.update(_extract_metadata(chapter.text, keywords=chapter.keywords, lexicon=lexicon))
        chapter.metadata.update(
            _extract_volume_metadata(
                chapter.title,
                chapter.text,
                source_hint=f"{loaded.relative_path} {loaded.path.name}",
            )
        )
        if enricher:
            chapter_enrichment = enricher.enrich_chapter(
                title=chapter.title,
                text=chapter.text,
                fallback_summary=chapter.summary,
                fallback_keywords=chapter.keywords,
                fallback_metadata=chapter.metadata,
            )
            if chapter_enrichment.summary:
                chapter.summary = chapter_enrichment.summary
            chapter.keywords = merge_keywords(chapter_enrichment.keywords, chapter.keywords, limit=KEYWORD_LIMIT)
            chapter.metadata = merge_metadata(base=chapter.metadata, enriched=chapter_enrichment.metadata)
            if chapter_enrichment.used_llm:
                chapter.metadata["llm_enriched"] = True

        local_chunks = _split_text(chapter.text)
        logger.info(
            f"Chapter {chapter.title}: split into {len(local_chunks)} candidate chunks."
        )
        chapter_chunk_ids: list[str] = []
        search_cursor = chapter.char_start

        # 阶段1:串行准备(字符定位 search_cursor 有状态,必须按序),只算本地字段,不发 LLM
        prepared_chunks: list[dict] = []
        for local_text in local_chunks:
            text = local_text.strip()
            if not text:
                continue

            char_start = _locate_chunk_start(loaded.normalized_text, text, search_cursor)
            char_end = char_start + len(text)
            search_cursor = max(char_start + 1, char_end - CHUNK_OVERLAP)

            chunk_id = _stable_id(
                "chunk",
                doc_id,
                chapter.chapter_id,
                str(char_start),
                str(char_end),
                _sha1_hex(text),
            )

            chunk_keywords = _extract_keywords(text, limit=8)
            chunk_metadata = _extract_metadata(text, keywords=chunk_keywords, lexicon=lexicon)
            prepared_chunks.append(
                {
                    "text": text,
                    "char_start": char_start,
                    "char_end": char_end,
                    "chunk_id": chunk_id,
                    "keywords": chunk_keywords,
                    "metadata": chunk_metadata,
                    "enrichment": None,
                }
            )

        # 阶段2:并发增强(每个 chunk 一次 LLM 调用,纯网络等待,用线程池并发;结果写回各自记录)
        if enricher and prepared_chunks:
            def _enrich_one(record: dict) -> None:
                record["enrichment"] = enricher.enrich_chunk(
                    chapter_title=chapter.title,
                    text=record["text"],
                    fallback_keywords=record["keywords"],
                    fallback_metadata=record["metadata"],
                )

            workers = max(1, min(PREPROCESS_CONCURRENCY, len(prepared_chunks)))
            logger.info(
                f"Chapter {chapter.title}: enriching {len(prepared_chunks)} chunks "
                f"with concurrency={workers}."
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_enrich_one, prepared_chunks))

        # 阶段3:串行组装(保持原顺序、chunk_index、占位符判断与相邻链接完全不变)
        total_prepared = len(prepared_chunks)
        for prepared_number, record in enumerate(prepared_chunks, start=1):
            text = record["text"]
            char_start = record["char_start"]
            char_end = record["char_end"]
            chunk_id = record["chunk_id"]
            chunk_keywords = record["keywords"]
            chunk_metadata = record["metadata"]

            chunk_enrichment = record["enrichment"]
            if chunk_enrichment is not None:
                chunk_keywords = merge_keywords(chunk_enrichment.keywords, chunk_keywords, limit=8)
                chunk_metadata = merge_metadata(base=chunk_metadata, enriched=chunk_enrichment.metadata)
                if chunk_enrichment.relations:
                    chunk_metadata["llm_relations"] = chunk_enrichment.relations
                if chunk_enrichment.used_llm:
                    chunk_metadata["llm_enriched"] = True
            chunk_metadata.update(
                {
                    "chapter_keywords": chapter.keywords,
                    "chapter_persons": chapter.metadata.get("persons", []),
                    "chapter_locations": chapter.metadata.get("locations", []),
                    "chapter_events": chapter.metadata.get("events", []),
                    "chapter_objects": chapter.metadata.get("objects", []),
                    "chapter_aliases": chapter.metadata.get("aliases", []),
                    "volume_label": chapter.metadata.get("volume_label"),
                    "volume_index": chapter.metadata.get("volume_index"),
                    "relative_path": loaded.relative_path,
                }
            )

            if _is_placeholder_chunk(chapter.title, text, chunk_metadata):
                chunk_metadata["is_placeholder_chunk"] = True
                logger.info(
                    f"Chapter {chapter.title}: skipped placeholder chunk "
                    f"{prepared_number}/{total_prepared}."
                )
                continue

            chunks.append(
                ChunkArtifact(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chapter_id=chapter.chapter_id,
                    source_path=str(loaded.path),
                    doc_name=loaded.path.name,
                    corpus_name=_infer_corpus_name(loaded.relative_path),
                    chapter_title=chapter.title,
                    chapter_index=chapter.chapter_index,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                    text=text,
                    summary=chapter.summary,
                    keywords=chunk_keywords,
                    metadata=chunk_metadata,
                )
            )
            chapter_chunk_ids.append(chunk_id)
            chunk_index += 1
            if prepared_number == total_prepared or prepared_number % 10 == 0:
                logger.info(
                    f"Chapter {chapter.title}: processed chunk {prepared_number}/{total_prepared}."
                )

        _link_adjacent_chunks(chunks, chapter_chunk_ids)
        logger.info(
            f"Chapter {chapter.title} complete: produced {len(chapter_chunk_ids)} chunks."
        )

    return chunks


def _extract_metadata(text: str, *, keywords: list[str], lexicon: EntityLexicon) -> dict[str, list[str]]:
    return {
        "persons": _extract_persons(text, lexicon)[:8],
        "locations": _extract_pattern_values(text, LOCATION_HINT_PATTERN)[:8],
        "events": _extract_pattern_values(text, EVENT_HINT_PATTERN)[:8],
        "objects": _extract_pattern_values(text, OBJECT_HINT_PATTERN)[:8],
        "aliases": _extract_aliases(text, lexicon)[:8],
        "keywords": keywords,
    }


def _build_role_index_chunks(
    loaded: LoadedText,
    *,
    doc_id: str,
    chapters: list[ChapterArtifact],
    chunks: list[ChunkArtifact],
    lexicon: EntityLexicon,
    enricher,
    starting_chunk_index: int,
) -> list[ChunkArtifact]:
    if not chunks:
        return []

    role_chunks: list[ChunkArtifact] = []
    role_index = starting_chunk_index

    doc_role_chunk = _build_single_role_index_chunk(
        loaded,
        doc_id=doc_id,
        chapter_id=_stable_id("chapter", doc_id, "synthetic", "角色总表"),
        chapter_title="角色总表",
        chapter_index=-1,
        scope_label="全文",
        source_chunks=chunks,
        lexicon=lexicon,
        enricher=enricher,
        chunk_index=role_index,
    )
    if doc_role_chunk:
        role_chunks.append(doc_role_chunk)
        role_index += 1

    by_volume: dict[tuple[str, int], list[ChunkArtifact]] = defaultdict(list)
    for chunk in chunks:
        label = str(chunk.metadata.get("volume_label", "") or "").strip()
        index = chunk.metadata.get("volume_index")
        if not label or index is None:
            continue
        try:
            volume_index = int(index)
        except Exception:
            continue
        by_volume[(label, volume_index)].append(chunk)

    for (volume_label, volume_index), volume_chunks in sorted(by_volume.items(), key=lambda item: item[0][1]):
        role_chunk = _build_single_role_index_chunk(
            loaded,
            doc_id=doc_id,
            chapter_id=_stable_id("chapter", doc_id, "synthetic", volume_label, "角色总表"),
            chapter_title=f"{volume_label}角色总表",
            chapter_index=-1,
            scope_label=volume_label,
            source_chunks=volume_chunks,
            lexicon=lexicon,
            enricher=enricher,
            chunk_index=role_index,
            volume_label=volume_label,
            volume_index=volume_index,
        )
        if role_chunk:
            role_chunks.append(role_chunk)
            role_index += 1

    return role_chunks


def _build_single_role_index_chunk(
    loaded: LoadedText,
    *,
    doc_id: str,
    chapter_id: str,
    chapter_title: str,
    chapter_index: int,
    scope_label: str,
    source_chunks: list[ChunkArtifact],
    lexicon: EntityLexicon,
    enricher,
    chunk_index: int,
    volume_label: str | None = None,
    volume_index: int | None = None,
) -> ChunkArtifact | None:
    person_counter: Counter[str] = Counter()
    relation_counter: Counter[str] = Counter()
    relation_type_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()

    evidence_parts: list[str] = []
    for chunk in source_chunks[:24]:
        for person in chunk.metadata.get("persons", []) or []:
            if person:
                person_counter[str(person)] += 1
        for relation in chunk.metadata.get("relations", []) or []:
            relation_counter[str(relation)] += 1
        for keyword in chunk.keywords[:6]:
            if keyword:
                keyword_counter[str(keyword)] += 1
        for relation in chunk.metadata.get("relations", []) or []:
            relation_text = str(relation)
            if ":" in relation_text:
                relation_type_counter[relation_text.split(":", 1)[-1]] += 1
        evidence_parts.append(
            f"[{chunk.chapter_title}] {re.sub(r'\\s+', ' ', chunk.text.strip())[:180]}"
        )

    if not person_counter:
        return None

    fallback_major = [name for name, _ in person_counter.most_common(12)]
    fallback_relationships = [name for name, _ in relation_counter.most_common(10)]
    fallback_keywords = [name for name, _ in keyword_counter.most_common(14)]
    evidence_text = "\n".join(evidence_parts[:18])

    role_summary = _build_role_index_summary(
        scope_label=scope_label,
        fallback_major=fallback_major,
        fallback_relationships=fallback_relationships,
    )
    female_candidates = _guess_gendered_characters(source_chunks, target="female")
    male_candidates = _guess_gendered_characters(source_chunks, target="male")
    llm_keywords: list[str] = []
    llm_major = fallback_major
    llm_relationships = fallback_relationships
    llm_summary = role_summary

    if enricher:
        role_enrichment = enricher.enrich_role_index(
            title=chapter_title,
            scope_label=scope_label,
            evidence_text=evidence_text,
            fallback_major_characters=fallback_major,
            fallback_relationships=fallback_relationships,
            fallback_keywords=fallback_keywords,
        )
        if role_enrichment.used_llm:
            if role_enrichment.summary:
                llm_summary = role_enrichment.summary
            llm_major = role_enrichment.major_characters or fallback_major
            female_candidates = role_enrichment.female_characters or female_candidates
            male_candidates = role_enrichment.male_characters or male_candidates
            llm_relationships = role_enrichment.important_relationships or fallback_relationships
            llm_keywords = role_enrichment.keywords

    text = _render_role_index_text(
        scope_label=scope_label,
        summary=llm_summary,
        major_characters=llm_major,
        female_characters=female_candidates,
        male_characters=male_candidates,
        relationships=llm_relationships,
    )
    chunk_keywords = merge_keywords(llm_keywords, fallback_keywords, llm_major, female_candidates, limit=18)
    chunk_metadata = {
        "persons": llm_major[:12],
        "chapter_persons": llm_major[:12],
        "locations": [],
        "events": [],
        "objects": [],
        "aliases": [],
        "keywords": chunk_keywords,
        "chapter_keywords": chunk_keywords,
        "chapter_locations": [],
        "chapter_events": [],
        "chapter_objects": [],
        "chapter_aliases": [],
        "relative_path": loaded.relative_path,
        "is_synthetic_role_index": True,
        "role_index_scope": scope_label,
        "female_characters": female_candidates[:10],
        "male_characters": male_candidates[:10],
        "important_relationships": llm_relationships[:12],
        "source_chunk_ids": [item.chunk_id for item in source_chunks[:24]],
        "role_relation_types": [name for name, _ in relation_type_counter.most_common(6)],
    }
    if volume_label and volume_index is not None:
        chunk_metadata["volume_label"] = volume_label
        chunk_metadata["volume_index"] = volume_index

    chunk_id = _stable_id(
        "chunk",
        doc_id,
        chapter_id,
        "role_index",
        scope_label,
        _sha1_hex(text),
    )
    return ChunkArtifact(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chapter_id=chapter_id,
        source_path=str(loaded.path),
        doc_name=loaded.path.name,
        corpus_name=_infer_corpus_name(loaded.relative_path),
        chapter_title=chapter_title,
        chapter_index=chapter_index,
        chunk_index=chunk_index,
        char_start=0,
        char_end=0,
        text=text,
        summary=llm_summary,
        keywords=chunk_keywords,
        metadata=chunk_metadata,
    )


def _build_role_index_summary(
    *,
    scope_label: str,
    fallback_major: list[str],
    fallback_relationships: list[str],
) -> str:
    major_text = "、".join(fallback_major[:6]) if fallback_major else "暂无明确角色"
    relation_text = "；关系线索：" + "、".join(fallback_relationships[:4]) if fallback_relationships else ""
    return f"{scope_label}角色总表：主要人物包括{major_text}{relation_text}。"


def _guess_gendered_characters(source_chunks: list[ChunkArtifact], *, target: str) -> list[str]:
    if target not in {"female", "male"}:
        return []
    cue = "她" if target == "female" else "他"
    counter: Counter[str] = Counter()
    for chunk in source_chunks:
        text = chunk.text
        if cue not in text:
            continue
        for person in chunk.metadata.get("persons", []) or []:
            if person and person in text:
                counter[str(person)] += text.count(cue)
    return [name for name, _ in counter.most_common(8)]


def _render_role_index_text(
    *,
    scope_label: str,
    summary: str,
    major_characters: list[str],
    female_characters: list[str],
    male_characters: list[str],
    relationships: list[str],
) -> str:
    sections = [
        f"【{scope_label}角色总表】",
        f"摘要：{summary}" if summary else "",
        f"主要角色：{'、'.join(major_characters) if major_characters else '暂无'}",
        f"女性角色：{'、'.join(female_characters) if female_characters else '暂无'}",
        f"男性角色：{'、'.join(male_characters) if male_characters else '暂无'}",
        f"重要关系：{'；'.join(relationships) if relationships else '暂无'}",
    ]
    return "\n".join(part for part in sections if part).strip()


def _extract_volume_metadata(title: str, text: str, source_hint: str = "") -> dict[str, object]:
    match = (
        _find_volume_label_and_index(title or "")
        or _find_volume_label_and_index(text[:160])
        or _find_volume_label_and_index(source_hint or "")
        or _find_parenthesized_volume_index(source_hint or "")
    )
    if not match:
        return {}
    label, index = match
    return {
        "volume_label": label,
        "volume_index": index,
    }


def _find_volume_label_and_index(text: str) -> tuple[str, int] | None:
    match = VOLUME_PATTERN.search(text or "")
    if not match:
        match = re.search(r"第\s*([0-9零〇一二两三四五六七八九十百]+)\s*(卷|册|部)", text or "")
    if not match:
        return None
    raw_number = match.group(1)
    unit = match.group(2)
    volume_index = _parse_volume_number(raw_number)
    if volume_index is None:
        return None
    return f"第{raw_number}{unit}", volume_index


def _find_parenthesized_volume_index(text: str) -> tuple[str, int] | None:
    match = re.search(r"\(([1-9][0-9]{0,2})\)", text or "")
    if not match:
        return None
    value = int(match.group(1))
    return f"第{value}卷", value


def _parse_volume_number(raw: str) -> int | None:
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    if cleaned.isdigit():
        value = int(cleaned)
        return value if value > 0 else None
    if cleaned == "十":
        return 10

    total = 0
    current = 0
    saw_unit = False
    for char in cleaned:
        if char in CN_NUMERAL_MAP:
            current = CN_NUMERAL_MAP[char]
            continue
        if char == "十":
            saw_unit = True
            total += (current or 1) * 10
            current = 0
            continue
        if char == "百":
            saw_unit = True
            total += (current or 1) * 100
            current = 0
            continue
        return None
    value = total + current
    if value <= 0 and saw_unit:
        return None
    return value or None


def _is_placeholder_chunk(chapter_title: str, text: str, metadata: dict[str, object]) -> bool:
    compact_title = re.sub(r"\s+", "", chapter_title or "")
    compact_text = re.sub(r"\s+", "", text or "")
    if len(compact_text) > 120:
        return False

    title_match = any(pattern.match(compact_title) for pattern in PLACEHOLDER_TITLE_PATTERNS)
    text_match = any(pattern.search(compact_text) for pattern in PLACEHOLDER_TEXT_PATTERNS)
    if not (title_match or text_match):
        return False

    entity_count = sum(
        len(metadata.get(key, []) or [])
        for key in ("persons", "locations", "events", "objects", "aliases")
    )
    if entity_count > 0:
        return False

    keyword_count = len(metadata.get("keywords", []) or [])
    return keyword_count <= 4


def _extract_persons(text: str, lexicon: EntityLexicon) -> list[str]:
    persons = []
    seen: set[str] = set()

    for candidate in _strong_person_candidates(text):
        normalized = _normalize_person_candidate(candidate)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        persons.append(normalized)

    for candidate in _raw_person_candidates(text):
        normalized = _normalize_person_candidate(candidate)
        if not normalized:
            continue
        if normalized not in lexicon.persons:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        persons.append(normalized)
    return persons


def _raw_person_candidates(text: str) -> list[str]:
    candidates = []
    candidates.extend(CN_NAME_PATTERN.findall(text))
    candidates.extend(EN_NAME_PATTERN.findall(text))
    candidates.extend(DOTTED_NAME_PATTERN.findall(text))
    return candidates


def _strong_person_candidates(text: str) -> list[str]:
    candidates = []
    candidates.extend(match[0] for match in TITLED_NAME_PATTERN.findall(text))
    candidates.extend(INTRO_NAME_PATTERN.findall(text))
    candidates.extend(DOTTED_NAME_PATTERN.findall(text))
    return candidates


def _titled_person_candidates(text: str) -> list[str]:
    return [match[0] for match in TITLED_NAME_PATTERN.findall(text)]


def _normalize_person_candidate(value: str) -> str:
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z·]+", "", value).strip()
    for prefix in NOISY_NAME_PREFIXES:
        if value.startswith(prefix) and len(value) > len(prefix) + 1:
            value = value[len(prefix) :]
            break
    for suffix in NOISY_NAME_SUFFIXES:
        if value.endswith(suffix) and len(value) > len(suffix) + 1:
            value = value[: -len(suffix)]
            break
    value = re.split(r"(先生|太太|夫人|教授|小姐|同学|女士|说道|说|问道|问|回答|看着|看|听到|来到|走到|在)", value, 1)[0]
    if len(value) < 2:
        return ""
    if value in STOPWORDS or value in GENERIC_ROLE_WORDS or value in NON_PERSON_TERMS:
        return ""
    if value.endswith(("谨上", "著", "编", "差", "小", "先", "本", "都", "拿", "捡", "巨")):
        return ""
    if any(fragment in value for fragment in BANNED_ENTITY_FRAGMENTS):
        return ""
    if value.endswith(BAD_NAME_ENDINGS):
        return ""
    if value.endswith(("吗", "吧", "了", "想", "又", "着", "时", "后", "中")):
        return ""
    if value.startswith(("叫", "说", "想", "问", "看", "听", "来", "走", "低", "立", "顿", "光")):
        return ""
    if value.isdigit():
        return ""
    if "·" in value:
        parts = value.split("·")
        if any(len(part) > 8 or len(part) < 1 for part in parts):
            return ""
        left, right = parts[0], parts[1]
        if all("\u4e00" <= char <= "\u9fff" for char in left):
            left = left[-4:]
        if all("\u4e00" <= char <= "\u9fff" for char in right):
            right = right[:4]
        value = f"{left}·{right}"
        if len(left) < 1 or len(right) < 1:
            return ""
        return value
    if all("\u4e00" <= char <= "\u9fff" for char in value):
        if value in {"波特"}:
            return ""
        if len(value) == 2:
            return value
        if len(value) in (3, 4) and (value[:2] in COMMON_SURNAME_PREFIXES or value[0] in "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"):
            return value
        return ""
    return value


def _has_person_like_context(
    sentence: str,
    raw_candidate: str,
    normalized_candidate: str,
    sentence_strong_candidates: set[str],
) -> bool:
    if normalized_candidate in sentence_strong_candidates or "·" in normalized_candidate:
        return True

    for strong_candidate in sentence_strong_candidates:
        if normalized_candidate and normalized_candidate in strong_candidate:
            return True

    for match in re.finditer(re.escape(raw_candidate), sentence):
        start, end = match.span()
        window = sentence[max(0, start - 4) : min(len(sentence), end + 4)]
        if NAME_CONTEXT_PATTERN.search(window):
            return True
    return False


def _should_keep_person_candidate(
    candidate: str,
    *,
    count: int,
    context_hit_count: int,
    strong_candidates: set[str],
) -> bool:
    if candidate in strong_candidates or "·" in candidate:
        return True
    if context_hit_count <= 0:
        return False
    if _looks_like_high_confidence_person(candidate):
        return count >= MIN_PERSON_FREQUENCY
    if all("\u4e00" <= char <= "\u9fff" for char in candidate) and len(candidate) == 2:
        return count >= 3
    return count >= max(MIN_PERSON_FREQUENCY + 1, 3)


def _extract_pattern_values(text: str, pattern: re.Pattern[str]) -> list[str]:
    values = []
    seen: set[str] = set()
    for match in pattern.findall(text):
        value = _clean_pattern_value(match)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _clean_pattern_value(value: str) -> str:
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", value).strip()
    if len(value) < 2:
        return ""
    for marker in LOCATION_SPLIT_MARKERS:
        if marker in value and len(value) > len(marker):
            value = value.split(marker)[-1]
    if any(fragment in value for fragment in BANNED_ENTITY_FRAGMENTS):
        return ""
    if value in STOPWORDS:
        return ""
    if len(value) > 8 and all("\u4e00" <= char <= "\u9fff" for char in value):
        return ""
    return value


def _extract_aliases(text: str, lexicon: EntityLexicon) -> list[str]:
    aliases = []
    seen: set[str] = set()
    for sentence in re.split(r"[。！？!?]\s*", text):
        if not sentence or not ALIAS_CONNECTOR_PATTERN.search(sentence):
            continue
        for candidate in _extract_persons(sentence, lexicon):
            if candidate in seen:
                continue
            seen.add(candidate)
            aliases.append(candidate)
    return aliases


def _build_relations(chunks: list[ChunkArtifact]) -> list[RelationArtifact]:
    grouped: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    relation_meta: dict[tuple[str, str, str, str, str], tuple[str, str, str, str, str, float]] = {}
    doc_lexicons = _build_relation_lexicons(chunks)

    for chunk in chunks:
        relation_labels = chunk.metadata.setdefault("relations", [])
        lexicon = doc_lexicons.get(chunk.doc_id)
        for sentence in re.split(r"[。！？!?]\s*", chunk.text):
            if not sentence:
                continue
            relation_type, confidence = _infer_relation(sentence)
            if not relation_type:
                continue
            sentence_persons = _relation_sentence_persons(sentence, chunk.metadata.get("persons", []), lexicon)
            pairs = _relation_pairs(sentence, sentence_persons, relation_type, lexicon=lexicon)
            if not pairs:
                continue
            for person_a, person_b in pairs:
                key = (
                    chunk.doc_id,
                    chunk.chapter_id,
                    *sorted((person_a, person_b)),
                    relation_type,
                )
                grouped[key].append(chunk.chunk_id)
                relation_meta[key] = (
                    chunk.source_path,
                    chunk.doc_name,
                    chunk.corpus_name,
                    person_a,
                    person_b,
                    confidence,
                )
                label = f"{person_a}-{person_b}:{relation_type}"
                if label not in relation_labels:
                    relation_labels.append(label)
        for relation in chunk.metadata.get("llm_relations", []) or []:
            person_a = str(relation.get("person_a", "")).strip()
            person_b = str(relation.get("person_b", "")).strip()
            relation_type = str(relation.get("relation_type", "")).strip()
            if not person_a or not person_b or not relation_type or person_a == person_b:
                continue
            key = (
                chunk.doc_id,
                chunk.chapter_id,
                *sorted((person_a, person_b)),
                relation_type,
            )
            grouped[key].append(chunk.chunk_id)
            relation_meta[key] = (
                chunk.source_path,
                chunk.doc_name,
                chunk.corpus_name,
                person_a,
                person_b,
                float(relation.get("confidence", 0.72) or 0.72),
            )
            label = f"{person_a}-{person_b}:{relation_type}"
            if label not in relation_labels:
                relation_labels.append(label)

    relations: list[RelationArtifact] = []
    for key, evidence_chunk_ids in grouped.items():
        doc_id, chapter_id, person_a, person_b, relation_type = key
        source_path, doc_name, corpus_name, left_name, right_name, confidence = relation_meta[key]
        relation_id = _stable_id(
            "rel",
            doc_id,
            chapter_id,
            person_a,
            person_b,
            relation_type,
        )
        relations.append(
            RelationArtifact(
                relation_id=relation_id,
                doc_id=doc_id,
                chapter_id=chapter_id,
                source_path=source_path,
                doc_name=doc_name,
                corpus_name=corpus_name,
                person_a=left_name,
                person_b=right_name,
                relation_type=relation_type,
                evidence_chunk_ids=sorted(set(evidence_chunk_ids)),
                confidence=confidence,
                metadata={"evidence_count": len(set(evidence_chunk_ids))},
            )
        )

    relations.sort(key=lambda item: (-item.confidence, item.person_a, item.person_b))
    return relations


def _infer_relation(text: str) -> tuple[str | None, float]:
    for relation_type, pattern in RELATION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        confidence = 0.78
        if len(re.sub(r"\s+", "", text)) <= 40:
            confidence += 0.05
        return relation_type, min(confidence, 0.9)
    return None, 0.0


def _build_relation_lexicons(chunks: list[ChunkArtifact]) -> dict[str, EntityLexicon]:
    per_doc_counter: dict[str, Counter[str]] = defaultdict(Counter)
    per_doc_strong: dict[str, set[str]] = defaultdict(set)
    per_doc_titled: dict[str, set[str]] = defaultdict(set)

    for chunk in chunks:
        for person in chunk.metadata.get("persons", []):
            per_doc_counter[chunk.doc_id][person] += 1
        for person in chunk.metadata.get("chapter_persons", []):
            per_doc_counter[chunk.doc_id][person] += 1
        for person in chunk.metadata.get("persons", [])[:3]:
            if _looks_like_high_confidence_person(person):
                per_doc_strong[chunk.doc_id].add(person)
        for candidate in _titled_person_candidates(chunk.text):
            normalized = _normalize_person_candidate(candidate)
            if normalized:
                per_doc_titled[chunk.doc_id].add(normalized)

    lexicons: dict[str, EntityLexicon] = {}
    for doc_id, counter in per_doc_counter.items():
        strong_persons = per_doc_strong.get(doc_id, set())
        titled_persons = per_doc_titled.get(doc_id, set())
        persons = {
            person
            for person, count in counter.items()
            if count >= MIN_RELATION_PERSON_FREQUENCY or person in strong_persons
        }
        lexicons[doc_id] = EntityLexicon(
            persons=persons,
            person_frequency=dict(counter),
            strong_persons=strong_persons,
            titled_persons=titled_persons,
        )
    return lexicons


def _relation_sentence_persons(
    sentence: str,
    chunk_persons: list[str],
    lexicon: EntityLexicon | None,
) -> list[str]:
    if lexicon is None:
        return []

    candidates = _extract_persons(sentence, lexicon)
    if not candidates:
        candidates = _persons_in_sentence(sentence, chunk_persons)

    filtered = []
    seen: set[str] = set()
    for person in candidates:
        if person in seen:
            continue
        if not _is_relation_ready_person(person, lexicon):
            continue
        seen.add(person)
        filtered.append(person)
    return filtered


def _persons_in_sentence(sentence: str, persons: list[str]) -> list[str]:
    hits = []
    seen: set[str] = set()
    for person in persons:
        if person in sentence and person not in seen:
            seen.add(person)
            hits.append(person)
    return hits


def _relation_pairs(
    sentence: str,
    persons: list[str],
    relation_type: str,
    lexicon: EntityLexicon | None = None,
) -> list[tuple[str, str]]:
    if len(persons) < 2:
        return []

    max_gap = 32 if relation_type in {"mentor", "helper", "enemy"} else 40
    pairs = []
    for left, right in _pairwise(persons[:4]):
        if not _is_relation_pair_valid(sentence, left, right, relation_type, max_gap=max_gap, lexicon=lexicon):
            continue
        pairs.append((left, right))
    return pairs


def _pairwise(values: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if left == right:
                continue
            pairs.append((left, right))
    return pairs


def _is_relation_ready_person(person: str, lexicon: EntityLexicon) -> bool:
    if person not in lexicon.persons:
        return False
    if len(person) > 12 or len(person) < 2:
        return False
    if person in STOPWORDS or person in GENERIC_ROLE_WORDS or person in NON_PERSON_TERMS:
        return False
    if any(fragment in person for fragment in BANNED_ENTITY_FRAGMENTS):
        return False
    if not _looks_like_high_confidence_person(person):
        count = lexicon.person_frequency.get(person, 0)
        if count < MIN_RELATION_PERSON_FREQUENCY:
            return False
    return True


def _looks_like_high_confidence_person(person: str) -> bool:
    if not person:
        return False
    if person in STOPWORDS or person in GENERIC_ROLE_WORDS or person in NON_PERSON_TERMS:
        return False
    if person.endswith(BAD_NAME_ENDINGS):
        return False
    if "路" in person:
        return True
    if all("\u4e00" <= char <= "\u9fff" for char in person):
        if len(person) == 2:
            return True
        if len(person) in (3, 4) and (
            person[:2] in COMMON_SURNAME_PREFIXES
            or person[0] in "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
        ):
            return True
        return False
    return bool(re.fullmatch(r"[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?", person))


def _is_relation_pair_valid(
    sentence: str,
    left: str,
    right: str,
    relation_type: str,
    *,
    max_gap: int,
    lexicon: EntityLexicon | None = None,
) -> bool:
    if left == right:
        return False
    if left in right or right in left:
        return False

    left_index = sentence.find(left)
    right_index = sentence.find(right)
    if left_index < 0 or right_index < 0:
        return False

    gap = abs(right_index - left_index)
    if gap > max_gap:
        return False

    between_start = min(left_index, right_index) + len(left)
    between_end = max(left_index, right_index)
    between = sentence[between_start:between_end]
    if len(re.sub(r"\s+", "", between)) > max_gap:
        return False

    trigger_spans = _relation_trigger_spans(sentence, relation_type)
    if not trigger_spans:
        return False

    pair_start = min(left_index, right_index)
    pair_end = max(left_index + len(left), right_index + len(right))
    if not any(_trigger_is_close_to_pair(pair_start, pair_end, trigger_start, trigger_end) for trigger_start, trigger_end in trigger_spans):
        return False

    if relation_type == "mentor":
        left_has_mentor_title = _has_title_near_person(sentence, left, MENTOR_TITLE_MARKERS)
        right_has_mentor_title = _has_title_near_person(sentence, right, MENTOR_TITLE_MARKERS)
        if not (left_has_mentor_title or right_has_mentor_title):
            return False
        if left_has_mentor_title and right_has_mentor_title:
            return False
        if not any(hint in sentence for hint in MENTOR_ACTION_HINTS):
            return False
        titled_person = left if left_has_mentor_title else right
        other_person = right if titled_person == left else left
        titled_index = sentence.find(titled_person)
        if titled_index < 0:
            return False
        if not _person_has_mention_after_index(sentence, other_person, titled_index):
            return False
        if _person_distance_to_nearest_trigger(sentence, other_person, trigger_spans) > 12:
            return False
        if lexicon is not None and other_person in lexicon.titled_persons:
            return False

    return True


def _relation_trigger_spans(sentence: str, relation_type: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for key, pattern in RELATION_PATTERNS:
        if key != relation_type:
            continue
        spans.extend(match.span() for match in pattern.finditer(sentence))
    return spans


def _trigger_is_close_to_pair(
    pair_start: int,
    pair_end: int,
    trigger_start: int,
    trigger_end: int,
) -> bool:
    if trigger_start <= pair_end and trigger_end >= pair_start:
        return True
    if abs(trigger_start - pair_end) <= 8:
        return True
    if abs(pair_start - trigger_end) <= 8:
        return True
    return False


def _has_title_near_person(sentence: str, person: str, titles: tuple[str, ...]) -> bool:
    for match in re.finditer(re.escape(person), sentence):
        start, end = match.span()
        window = sentence[max(0, start - 2) : min(len(sentence), end + 3)]
        if any(title in window for title in titles):
            return True
    return False


def _person_distance_to_nearest_trigger(
    sentence: str,
    person: str,
    trigger_spans: list[tuple[int, int]],
) -> int:
    positions = [match.span() for match in re.finditer(re.escape(person), sentence)]
    if not positions or not trigger_spans:
        return 10**9

    min_distance = 10**9
    for person_start, person_end in positions:
        for trigger_start, trigger_end in trigger_spans:
            if trigger_start <= person_end and trigger_end >= person_start:
                return 0
            min_distance = min(
                min_distance,
                abs(trigger_start - person_end),
                abs(person_start - trigger_end),
            )
    return min_distance


def _person_has_mention_after_index(sentence: str, person: str, start_index: int) -> bool:
    for match in re.finditer(re.escape(person), sentence):
        if match.start() >= start_index:
            return True
    return False


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    segments = _segment_text(text)
    chunks: list[str] = []
    current = ""

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if not current:
            current = segment
            continue
        if len(current) + len(segment) <= CHUNK_SIZE:
            current += segment
            continue
        chunks.append(current.strip())
        overlap = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else ""
        current = f"{overlap}{segment}"
        if len(current) > CHUNK_SIZE:
            forced = _force_split(current)
            chunks.extend(forced[:-1])
            current = forced[-1] if forced else ""

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _segment_text(text: str) -> list[str]:
    parts = re.split(r"(\n\n+|\n|[。！？；!?])", text)
    if len(parts) == 1:
        return _force_split(text)

    segments: list[str] = []
    cursor = ""
    for part in parts:
        if not part:
            continue
        cursor += part
        if part.startswith("\n") or re.fullmatch(r"[。！？；!?]", part):
            segments.append(cursor)
            cursor = ""

    if cursor:
        segments.append(cursor)

    normalized: list[str] = []
    for segment in segments:
        if len(segment) <= CHUNK_SIZE:
            normalized.append(segment)
            continue
        normalized.extend(_force_split(segment))
    return normalized


def _force_split(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _build_chapter_artifact(
    loaded: LoadedText,
    *,
    doc_id: str | None,
    chapter_index: int,
    title: str,
    char_start: int,
    char_end: int,
    text: str,
) -> ChapterArtifact:
    chapter_id = _stable_id(
        "chapter",
        doc_id or "pending",
        str(chapter_index),
        title,
        str(char_start),
        str(char_end),
    )
    return ChapterArtifact(
        chapter_id=chapter_id,
        doc_id=doc_id or "",
        source_path=str(loaded.path),
        doc_name=loaded.path.name,
        corpus_name=_infer_corpus_name(loaded.relative_path),
        title=_normalize_title(title),
        chapter_index=chapter_index,
        char_start=char_start,
        char_end=char_end,
        text=text,
        summary="",
        keywords=[],
        metadata={"relative_path": loaded.relative_path},
    )


def _is_chapter_heading(text: str) -> bool:
    if len(text) > 80:
        return False
    return any(pattern.match(text) for pattern in CHAPTER_PATTERNS)


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip() or DEFAULT_CHAPTER_TITLE


def _extract_keywords(text: str, *, limit: int) -> list[str]:
    tokens = [token for token in TOKEN_PATTERN.findall(text) if len(token.strip()) >= 2]
    if not tokens:
        return []

    frequencies = Counter(token.lower() for token in tokens)
    ranked = sorted(
        frequencies.items(),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
    return [token for token, _ in ranked[:limit]]


def _build_summary(text: str, *, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized

    sentences = re.split(r"(?<=[。！？!?])", normalized)
    summary: list[str] = []
    total = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        next_total = total + len(sentence)
        if summary and next_total > limit:
            break
        summary.append(sentence)
        total = next_total
        if total >= limit:
            break
    if summary:
        return "".join(summary)[:limit].strip()
    return normalized[:limit].strip()


def _locate_chunk_start(full_text: str, chunk_text: str, hint_start: int) -> int:
    offset = full_text.find(chunk_text, hint_start)
    if offset >= 0:
        return offset

    fallback = full_text.find(chunk_text)
    if fallback >= 0:
        return fallback

    return hint_start


def _link_adjacent_chunks(chunks: list[ChunkArtifact], chapter_chunk_ids: list[str]) -> None:
    if not chapter_chunk_ids:
        return

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for index, chunk_id in enumerate(chapter_chunk_ids):
        chunk = by_id[chunk_id]
        chunk.prev_chunk_id = chapter_chunk_ids[index - 1] if index > 0 else None
        chunk.next_chunk_id = chapter_chunk_ids[index + 1] if index + 1 < len(chapter_chunk_ids) else None


def _common_base_dir(file_paths: list[Path]) -> Path:
    resolved = [str(path.resolve()) for path in file_paths]
    return Path(os.path.commonpath(resolved))


def _stable_id(prefix: str, *parts: str) -> str:
    joined = "||".join(parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _infer_corpus_name(relative_path: str) -> str:
    normalized = str(relative_path or "").replace("\\", "/").strip("/")
    if not normalized:
        return "default"
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return parts[0]
    stem = Path(parts[0]).stem if parts else "default"
    return stem or "default"
