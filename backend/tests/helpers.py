"""
Helper functions for building mock objects in tests.
These can be imported directly unlike conftest.py fixtures.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults


def build_search_results(
    documents: list[str],
    metadata: list[dict[str, Any]],
    distances: list[float] | None = None,
    error: str | None = None,
) -> SearchResults:
    """Create SearchResults for testing."""
    if distances is None:
        distances = [0.5] * len(documents)
    return SearchResults(
        documents=documents, metadata=metadata, distances=distances, error=error
    )


def build_empty_search_results(error_msg: str = None) -> SearchResults:
    """Create empty SearchResults."""
    return (
        SearchResults.empty(error_msg)
        if error_msg
        else SearchResults(documents=[], metadata=[], distances=[])
    )


def build_chroma_results(
    documents: list[str],
    metadata: list[dict[str, Any]],
    distances: list[float] | None = None,
) -> dict:
    """Build mock ChromaDB query results format."""
    if distances is None:
        distances = [0.5] * len(documents)
    return {"documents": [documents], "metadatas": [metadata], "distances": [distances]}


def build_claude_text_response(text: str) -> Mock:
    """Create mock Claude response with text content (no tool use)."""
    text_block = Mock()
    text_block.type = "text"
    text_block.text = text

    response = Mock()
    response.stop_reason = "end_turn"
    response.content = [text_block]

    return response


def build_claude_tool_use_response(
    tool_name: str, tool_input: dict[str, Any], tool_id: str = "tool_123"
) -> Mock:
    """Create mock Claude response requesting tool use."""
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_id

    response = Mock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]

    return response


def build_claude_mixed_response(
    text: str, tool_name: str, tool_input: dict[str, Any], tool_id: str = "tool_123"
) -> Mock:
    """Create mock Claude response with both text and tool use."""
    text_block = Mock()
    text_block.type = "text"
    text_block.text = text

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.input = tool_input
    tool_block.id = tool_id

    response = Mock()
    response.stop_reason = "tool_use"
    response.content = [text_block, tool_block]

    return response
