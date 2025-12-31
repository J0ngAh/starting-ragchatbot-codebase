"""
Integration tests for RAGSystem.query() method.

Tests evaluate:
1. Session handling
2. Tool definition passing
3. Source tracking and retrieval
4. Conversation history management
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create all mocked dependencies for RAGSystem."""
        with patch('rag_system.DocumentProcessor') as mock_dp, \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm, \
             patch('rag_system.ToolManager') as mock_tm, \
             patch('rag_system.CourseSearchTool') as mock_cst, \
             patch('rag_system.CourseOutlineTool') as mock_cot:

            # Setup mock instances
            mock_dp_instance = MagicMock()
            mock_vs_instance = MagicMock()
            mock_ai_instance = MagicMock()
            mock_sm_instance = MagicMock()
            mock_tm_instance = MagicMock()
            mock_cst_instance = MagicMock()
            mock_cot_instance = MagicMock()

            mock_dp.return_value = mock_dp_instance
            mock_vs.return_value = mock_vs_instance
            mock_ai.return_value = mock_ai_instance
            mock_sm.return_value = mock_sm_instance
            mock_tm.return_value = mock_tm_instance
            mock_cst.return_value = mock_cst_instance
            mock_cot.return_value = mock_cot_instance

            # Default behaviors
            mock_ai_instance.generate_response.return_value = "AI response"
            mock_sm_instance.get_conversation_history.return_value = None
            mock_tm_instance.get_tool_definitions.return_value = []
            mock_tm_instance.get_last_sources.return_value = []

            yield {
                'document_processor': mock_dp_instance,
                'vector_store': mock_vs_instance,
                'ai_generator': mock_ai_instance,
                'session_manager': mock_sm_instance,
                'tool_manager': mock_tm_instance,
                'search_tool': mock_cst_instance,
                'outline_tool': mock_cot_instance,
            }

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "test-model"
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "test-model"
        config.MAX_RESULTS = 5
        config.MAX_HISTORY = 2
        return config

    def test_query_retrieves_history_for_session(self, mock_dependencies, mock_config):
        """Verify session history is retrieved when session_id provided."""
        from rag_system import RAGSystem

        mock_dependencies['session_manager'].get_conversation_history.return_value = (
            "User: Previous\nAssistant: Reply"
        )

        rag = RAGSystem(mock_config)
        rag.query("New question", session_id="session_123")

        mock_dependencies['session_manager'].get_conversation_history.assert_called_with(
            "session_123"
        )

    def test_query_passes_history_to_ai_generator(self, mock_dependencies, mock_config):
        """Verify conversation history is passed to AI generator."""
        from rag_system import RAGSystem

        history = "User: Hello\nAssistant: Hi!"
        mock_dependencies['session_manager'].get_conversation_history.return_value = history

        rag = RAGSystem(mock_config)
        rag.query("Follow up", session_id="session_1")

        call_kwargs = mock_dependencies['ai_generator'].generate_response.call_args.kwargs
        assert call_kwargs['conversation_history'] == history

    def test_query_passes_tools_to_ai_generator(self, mock_dependencies, mock_config):
        """Verify tool definitions are passed to AI generator."""
        from rag_system import RAGSystem

        tool_defs = [{"name": "search", "description": "Search tool", "input_schema": {}}]
        mock_dependencies['tool_manager'].get_tool_definitions.return_value = tool_defs

        rag = RAGSystem(mock_config)
        rag.query("Test query")

        call_kwargs = mock_dependencies['ai_generator'].generate_response.call_args.kwargs
        assert call_kwargs['tools'] == tool_defs

    def test_query_passes_tool_manager_to_ai_generator(self, mock_dependencies, mock_config):
        """Verify tool manager is passed for execution."""
        from rag_system import RAGSystem

        rag = RAGSystem(mock_config)
        rag.query("Test query")

        call_kwargs = mock_dependencies['ai_generator'].generate_response.call_args.kwargs
        assert 'tool_manager' in call_kwargs

    def test_query_retrieves_sources_after_generation(self, mock_dependencies, mock_config):
        """Verify sources are retrieved from tool manager."""
        from rag_system import RAGSystem

        expected_sources = [{"title": "Course", "lesson": 1, "url": "http://test"}]
        mock_dependencies['tool_manager'].get_last_sources.return_value = expected_sources

        rag = RAGSystem(mock_config)
        response, sources = rag.query("Test query")

        mock_dependencies['tool_manager'].get_last_sources.assert_called_once()
        assert sources == expected_sources

    def test_query_resets_sources_after_retrieval(self, mock_dependencies, mock_config):
        """Verify sources are reset after being retrieved."""
        from rag_system import RAGSystem

        rag = RAGSystem(mock_config)
        rag.query("Test query")

        mock_dependencies['tool_manager'].reset_sources.assert_called_once()

    def test_query_updates_conversation_history(self, mock_dependencies, mock_config):
        """Verify conversation history is updated after query."""
        from rag_system import RAGSystem

        mock_dependencies['ai_generator'].generate_response.return_value = "AI answer"

        rag = RAGSystem(mock_config)
        rag.query("User question", session_id="session_1")

        mock_dependencies['session_manager'].add_exchange.assert_called_with(
            "session_1",
            "User question",
            "AI answer"
        )

    def test_query_does_not_update_history_without_session(self, mock_dependencies, mock_config):
        """Verify history not updated when no session_id."""
        from rag_system import RAGSystem

        rag = RAGSystem(mock_config)
        rag.query("Question", session_id=None)

        mock_dependencies['session_manager'].add_exchange.assert_not_called()

    def test_query_returns_response_and_sources(self, mock_dependencies, mock_config):
        """Verify query returns tuple of (response, sources)."""
        from rag_system import RAGSystem

        mock_dependencies['ai_generator'].generate_response.return_value = "The answer"
        mock_dependencies['tool_manager'].get_last_sources.return_value = [
            {"title": "ML Course", "lesson": 1, "url": "http://ml.com/1"}
        ]

        rag = RAGSystem(mock_config)
        response, sources = rag.query("What is ML?")

        assert response == "The answer"
        assert len(sources) == 1
        assert sources[0]["title"] == "ML Course"

    def test_query_returns_empty_sources_when_none(self, mock_dependencies, mock_config):
        """Verify empty list returned when no sources."""
        from rag_system import RAGSystem

        mock_dependencies['tool_manager'].get_last_sources.return_value = []

        rag = RAGSystem(mock_config)
        response, sources = rag.query("General question")

        assert sources == []

    def test_query_includes_prompt_wrapper(self, mock_dependencies, mock_config):
        """Verify query is wrapped in prompt template."""
        from rag_system import RAGSystem

        rag = RAGSystem(mock_config)
        rag.query("What is Python?")

        call_kwargs = mock_dependencies['ai_generator'].generate_response.call_args.kwargs
        query = call_kwargs['query']

        # The prompt wraps the user query
        assert "What is Python?" in query
        assert "course materials" in query.lower()


class TestRAGSystemInitialization:
    """Tests for RAGSystem initialization."""

    def test_initialization_creates_all_components(self):
        """Verify all components are created during init."""
        with patch('rag_system.DocumentProcessor') as mock_dp, \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_sm, \
             patch('rag_system.ToolManager') as mock_tm, \
             patch('rag_system.CourseSearchTool') as mock_cst, \
             patch('rag_system.CourseOutlineTool') as mock_cot:

            from rag_system import RAGSystem

            config = MagicMock()
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test"
            config.EMBEDDING_MODEL = "model"
            config.ANTHROPIC_API_KEY = "key"
            config.ANTHROPIC_MODEL = "model"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2

            rag = RAGSystem(config)

            mock_dp.assert_called_once()
            mock_vs.assert_called_once()
            mock_ai.assert_called_once()
            mock_sm.assert_called_once()
            mock_tm.assert_called_once()

    def test_initialization_registers_both_tools(self):
        """Verify both search and outline tools are registered."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager') as mock_tm, \
             patch('rag_system.CourseSearchTool') as mock_cst, \
             patch('rag_system.CourseOutlineTool') as mock_cot:

            from rag_system import RAGSystem

            config = MagicMock()
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.CHROMA_PATH = "./test"
            config.EMBEDDING_MODEL = "model"
            config.ANTHROPIC_API_KEY = "key"
            config.ANTHROPIC_MODEL = "model"
            config.MAX_RESULTS = 5
            config.MAX_HISTORY = 2

            rag = RAGSystem(config)

            # Both tools should be registered
            tm_instance = mock_tm.return_value
            assert tm_instance.register_tool.call_count == 2
