# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot for querying course materials. Users ask natural language questions and receive AI-powered answers with source citations.

## Commands

**Always use `uv` to manage dependencies and run Python files (e.g., `uv run python script.py`). Do not use `pip` directly.**

```bash
# Install dependencies
uv sync

# Install dev dependencies (includes black, isort, ruff)
uv sync --extra dev

# Run the application (starts server at http://localhost:8000)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# API docs available at http://localhost:8000/docs

# Code Quality
./scripts/format.sh   # Format code with black and isort
./scripts/lint.sh     # Run linting checks (ruff, black --check, isort --check)
./scripts/quality.sh  # Run all quality checks (format + lint + tests)
```

## Architecture

### Query Flow

```
Frontend (script.js) → FastAPI (app.py) → RAGSystem (rag_system.py)
    → AIGenerator calls Claude with tools
    → Claude decides to use search_course_content tool
    → CourseSearchTool queries VectorStore
    → VectorStore searches ChromaDB (semantic search)
    → Results returned to Claude for synthesis
    → Response + sources sent to frontend
```

### Backend Components (`backend/`)

| File | Purpose |
|------|---------|
| `rag_system.py` | Main orchestrator - coordinates all components |
| `ai_generator.py` | Claude API integration with tool execution loop |
| `vector_store.py` | ChromaDB interface with two collections: `course_catalog` (metadata) and `course_content` (chunks) |
| `document_processor.py` | Parses course files, extracts metadata, chunks text with overlap |
| `search_tools.py` | Tool definitions for Claude tool-use; `CourseSearchTool` implements the search |
| `session_manager.py` | Conversation history per session |
| `config.py` | Centralized settings (chunk size, models, paths) |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |

### Key Design Patterns

- **Tool-based AI**: Claude autonomously decides when to search using the `search_course_content` tool
- **Two-collection vector store**: `course_catalog` for semantic course name resolution, `course_content` for content search
- **Sentence-aware chunking**: Documents split at sentence boundaries with configurable overlap (default: 800 chars, 100 overlap)

## Configuration (`backend/config.py`)

- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2 (via sentence-transformers)
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `CHROMA_PATH`: ./chroma_db

## Course Document Format

Files in `docs/` folder (`.txt`, `.pdf`, `.docx`) should follow:

```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[content...]

Lesson 1: [lesson title]
[content...]
```

Documents are automatically loaded on server startup.

## Environment

Requires `.env` file with:
```
ANTHROPIC_API_KEY=sk-ant-xxxxx
```
