"""
Shared fixtures and mock builders for RAG chatbot tests.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


# =============================================================================
# Mock Response Builders
# =============================================================================

def build_search_results(
    documents: List[str],
    metadata: List[Dict[str, Any]],
    distances: Optional[List[float]] = None,
    error: Optional[str] = None
) -> SearchResults:
    """Create SearchResults for testing."""
    if distances is None:
        distances = [0.5] * len(documents)
    return SearchResults(
        documents=documents,
        metadata=metadata,
        distances=distances,
        error=error
    )


def build_empty_search_results(error_msg: str = None) -> SearchResults:
    """Create empty SearchResults."""
    return SearchResults.empty(error_msg) if error_msg else SearchResults(
        documents=[], metadata=[], distances=[]
    )


def build_chroma_results(
    documents: List[str],
    metadata: List[Dict[str, Any]],
    distances: Optional[List[float]] = None
) -> Dict:
    """Build mock ChromaDB query results format."""
    if distances is None:
        distances = [0.5] * len(documents)
    return {
        "documents": [documents],
        "metadatas": [metadata],
        "distances": [distances]
    }


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
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_id: str = "tool_123"
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
    text: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_id: str = "tool_123"
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


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_course():
    """Sample Course object for testing."""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://example.com/ml/0"
            ),
            Lesson(
                lesson_number=1,
                title="Supervised Learning",
                lesson_link="https://example.com/ml/1"
            ),
            Lesson(
                lesson_number=2,
                title="Neural Networks",
                lesson_link="https://example.com/ml/2"
            ),
        ]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample CourseChunk list for testing."""
    return [
        CourseChunk(
            content="This is content about machine learning basics.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning involves labeled data.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Neural networks are inspired by the brain.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2
        ),
    ]


@pytest.fixture
def sample_search_metadata():
    """Sample metadata for search results."""
    return [
        {"course_title": "Introduction to ML", "lesson_number": 1, "chunk_index": 0},
        {"course_title": "Introduction to ML", "lesson_number": 2, "chunk_index": 1},
    ]


@pytest.fixture
def sample_search_documents():
    """Sample document content for search results."""
    return [
        "This section covers neural networks and deep learning concepts.",
        "Machine learning algorithms can be categorized into supervised and unsupervised.",
    ]


# =============================================================================
# Mock VectorStore Fixture
# =============================================================================

@pytest.fixture
def mock_vector_store(sample_search_documents, sample_search_metadata):
    """Mock VectorStore with configurable search results."""
    mock_store = MagicMock()

    # Default search behavior - returns sample results
    default_results = build_search_results(
        documents=sample_search_documents,
        metadata=sample_search_metadata,
        distances=[0.3, 0.5]
    )
    mock_store.search.return_value = default_results

    # Default lesson link behavior
    mock_store.get_lesson_link.return_value = "https://example.com/lesson/1"

    # Default course outline behavior
    mock_store.get_course_outline.return_value = {
        "title": "Introduction to ML",
        "course_link": "https://example.com/ml",
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "https://example.com/ml/0"},
            {"lesson_number": 1, "lesson_title": "Basics", "lesson_link": "https://example.com/ml/1"},
        ]
    }

    return mock_store


@pytest.fixture
def mock_vector_store_empty():
    """Mock VectorStore that returns empty results."""
    mock_store = MagicMock()
    mock_store.search.return_value = build_search_results(
        documents=[], metadata=[], distances=[]
    )
    mock_store.get_lesson_link.return_value = None
    mock_store.get_course_outline.return_value = None
    return mock_store


@pytest.fixture
def mock_vector_store_error():
    """Mock VectorStore that returns error results."""
    mock_store = MagicMock()
    mock_store.search.return_value = SearchResults.empty("Search error: connection failed")
    return mock_store


# =============================================================================
# Mock Anthropic Client Fixture
# =============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client with configurable responses."""
    mock_client = MagicMock()

    # Default behavior - returns direct text response
    default_response = build_claude_text_response("This is a helpful response.")
    mock_client.messages.create.return_value = default_response

    return mock_client


@pytest.fixture
def mock_anthropic_client_tool_use():
    """Mock Anthropic client that triggers tool use then returns final response."""
    mock_client = MagicMock()

    # First call triggers tool use
    tool_response = build_claude_tool_use_response(
        tool_name="search_course_content",
        tool_input={"query": "neural networks"},
        tool_id="tool_abc123"
    )

    # Second call (after tool execution) returns final text
    final_response = build_claude_text_response(
        "Based on the course content, neural networks are a key topic covered in the ML course."
    )

    mock_client.messages.create.side_effect = [tool_response, final_response]

    return mock_client


# =============================================================================
# Mock Config Fixture
# =============================================================================

@pytest.fixture
def mock_config():
    """Test configuration with dummy values."""
    config = MagicMock()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


# =============================================================================
# Tool Manager Fixtures
# =============================================================================

@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing."""
    mock_tm = MagicMock()
    mock_tm.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ]
    mock_tm.execute_tool.return_value = "[Introduction to ML - Lesson 1]\nThis is about neural networks."
    mock_tm.get_last_sources.return_value = [
        {"title": "Introduction to ML", "lesson": 1, "url": "https://example.com/ml/1"}
    ]
    mock_tm.reset_sources.return_value = None
    return mock_tm


# =============================================================================
# Session Manager Fixtures
# =============================================================================

@pytest.fixture
def mock_session_manager():
    """Mock SessionManager for testing."""
    mock_sm = MagicMock()
    mock_sm.create_session.return_value = "session_1"
    mock_sm.get_conversation_history.return_value = "User: Hello\nAssistant: Hi there!"
    mock_sm.add_exchange.return_value = None
    return mock_sm
