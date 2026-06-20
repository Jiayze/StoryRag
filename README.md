# StoryRAG

StoryRAG is a local desktop RAG app for Chinese fiction and long-form text. It builds a local Chroma knowledge base from `.txt` files, enriches retrieval with metadata and optional LLM preprocessing, and answers questions with evidence-grounded citations and guardrails.

The project is designed for personal reading/research workflows: import text, build an index, ask questions, inspect evidence, pin context for follow-up questions, and export/import knowledge packages.

## Features

- Desktop UI built with PySide6.
- Local document preprocessing with chapter splitting, chunking, entity extraction, relation hints, and synthetic role index chunks.
- Hybrid retrieval over dense vectors, lexical keywords, metadata, relation signals, volume hints, and adjacent-context expansion.
- Optional DeepSeek-compatible LLM calls for preprocessing, query enrichment, decomposition planning, and answer generation.
- SiliconFlow/OpenAI-compatible embedding client with batching, concurrency, timeout, and retry controls.
- Multi-corpus import, incremental update, alias management, package export/import, and debug evidence views.

## Repository Contents

- `desktop_app/` - PySide6 desktop UI.
- `knowledge_base/` - corpus import, preprocessing orchestration, and Chroma writes.
- `preprocessing/` - text loading, chunking, metadata extraction, and artifact persistence.
- `retrieval/` - query analysis, hybrid ranking, vector store access, context expansion, and formatting.
- `qa/` - question answering flow, optional decomposition, and follow-up evidence payloads.
- `llm/` - OpenAI-compatible clients, prompts, response schema, and grounding validation.
- `tests/` - strategy and pure-function regression tests.

## Requirements

- Python 3.11, 3.12, or 3.13.
- DeepSeek-compatible chat API key for answer generation and optional LLM preprocessing.
- SiliconFlow-compatible embedding API key for building and querying the vector index.

Install runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For development and tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Configuration

Create a local `.env` from `.env.example` and fill in your own keys:

```powershell
Copy-Item .env.example .env
```

Required values:

```dotenv
DEEPSEEK_API_KEY=
SILICONFLOW_API_KEY=
```

Common optional values:

```dotenv
DEEPSEEK_MODEL=dsv4pro
DEEPSEEK_API_BASE=https://api.deepseek.com
RAG_EMBEDDING_MODEL=BAAI/bge-m3
SILICONFLOW_API_BASE=https://api.siliconflow.cn/v1
RAG_DOC_DIR=docs
RAG_PROCESSED_DIR=processed
RAG_CHROMA_DB_DIR=chroma_db
```

The desktop settings dialog can also write these values to the local `.env`.

## Usage

Run the desktop app:

```powershell
.\.venv\Scripts\python.exe run_desktop.py
```

Build from existing `.txt` files in `docs/`:

```powershell
.\.venv\Scripts\python.exe build_db.py
```

Build a Windows desktop executable:

```powershell
.\.venv\Scripts\python.exe build_desktop_exe.py
```

Typical desktop workflow:

1. Start the app.
2. Open settings and configure API keys/models.
3. Import local `.txt` files into a new or existing knowledge base.
4. Ask a question after indexing completes.
5. Inspect retrieved evidence, pin useful context, and continue follow-up questions.
6. Export/import knowledge packages when you need to move a built corpus between machines.

## Development

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run static checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall app.py app_services.py build_db.py build_desktop_exe.py core desktop_app knowledge_base llm preprocessing qa retrieval run_desktop.py
```

The CI workflow runs the same checks in a clean environment.

## Data And Copyright

This repository intentionally does not include source novels, copyrighted text, generated Chroma databases, preprocessing artifacts, caches, local virtual environments, IDE state, or API keys.

Ignored local data includes:

- `.env`
- `docs/`
- `processed/`
- `chroma_db/`
- `cache/`
- `.venv/`

Only import text that you own, have permission to process, or are legally allowed to use. Knowledge packages exported by the app may contain source text and embeddings, so treat them as private unless you have the rights to share their contents.

## Notes

- Simple factual questions skip the decomposition planner LLM; marker-based multi-part questions such as "只有", "分别", "哪些", and "有没有去过" still use the planner.
- Adjacent context expansion remains enabled because narrative RAG often depends on immediate surrounding text.
- The app currently uses some Chroma private collection APIs internally. This works for the pinned dependency set, but future hardening should wrap those accesses behind a repository adapter.
