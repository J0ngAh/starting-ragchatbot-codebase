"""
Integration tests for end-to-end tool execution flow.

Tests evaluate:
1. Complete search flow: AI -> ToolManager -> Tool -> VectorStore
2. Tool results passing back to Claude
3. Source population after searches
4. Direct response path (no tool use)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest

# Add backend and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from vector_store import SearchResults
from helpers import (
    build_search_results,
    build_claude_text_response,
    build_claude_tool_use_response
)


class TestSearchToolExecutionFlow:
    """Tests for complete search tool execution flow."""

    def test_search_tool_triggered_for_course_question(self):
        """
        Verify complete flow when Claude decides to use search_course_content.

        Flow: AIGenerator -> ToolManager -> CourseSearchTool -> VectorStore
        """
        # Setup mock VectorStore
        mock_vs = MagicMock()
        mock_vs.search.return_value = build_search_results(
            documents=["Neural networks are machine learning models."],
            metadata=[{"course_title": "ML Basics", "lesson_number": 1}]
        )
        mock_vs.get_lesson_link.return_value = "https://example.com/ml/1"

        # Setup ToolManager with real CourseSearchTool
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        # Execute tool through manager (simulating what AIGenerator does)
        result = tool_manager.execute_tool(
            "search_course_content",
            query="neural networks"
        )

        # Verify VectorStore.search was called
        mock_vs.search.assert_called_once_with(
            query="neural networks",
            course_name=None,
            lesson_number=None
        )

        # Verify result is formatted correctly
        assert "[ML Basics - Lesson 1]" in result
        assert "Neural networks" in result

        # Verify sources are tracked
        sources = tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["title"] == "ML Basics"
        assert sources[0]["url"] == "https://example.com/ml/1"

    def test_search_tool_with_course_filter(self):
        """Verify course filter is passed through the flow."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = build_search_results(
            documents=["Python content here."],
            metadata=[{"course_title": "Python 101", "lesson_number": 2}]
        )
        mock_vs.get_lesson_link.return_value = "https://example.com/py/2"

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        result = tool_manager.execute_tool(
            "search_course_content",
            query="basics",
            course_name="Python"
        )

        # Verify course filter was passed
        mock_vs.search.assert_called_once_with(
            query="basics",
            course_name="Python",
            lesson_number=None
        )

    def test_search_tool_with_lesson_filter(self):
        """Verify lesson filter is passed through the flow."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = build_search_results(
            documents=["Lesson 3 content."],
            metadata=[{"course_title": "Course", "lesson_number": 3}]
        )
        mock_vs.get_lesson_link.return_value = "https://example.com/3"

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        tool_manager.execute_tool(
            "search_course_content",
            query="test",
            lesson_number=3
        )

        call_kwargs = mock_vs.search.call_args.kwargs
        assert call_kwargs["lesson_number"] == 3


class TestOutlineToolExecutionFlow:
    """Tests for course outline tool execution flow."""

    def test_outline_tool_returns_course_structure(self):
        """Verify outline tool retrieves and formats course structure."""
        mock_vs = MagicMock()
        mock_vs.get_course_outline.return_value = {
            "title": "Machine Learning Course",
            "course_link": "https://example.com/ml",
            "lessons": [
                {"lesson_number": 0, "lesson_title": "Introduction", "lesson_link": "http://1"},
                {"lesson_number": 1, "lesson_title": "Basics", "lesson_link": "http://2"},
            ]
        }

        tool_manager = ToolManager()
        outline_tool = CourseOutlineTool(mock_vs)
        tool_manager.register_tool(outline_tool)

        result = tool_manager.execute_tool(
            "get_course_outline",
            course_name="ML"
        )

        mock_vs.get_course_outline.assert_called_once_with("ML")
        assert "Machine Learning Course" in result
        assert "Lesson 0: Introduction" in result
        assert "Lesson 1: Basics" in result

    def test_outline_tool_handles_not_found(self):
        """Verify outline tool handles course not found."""
        mock_vs = MagicMock()
        mock_vs.get_course_outline.return_value = None

        tool_manager = ToolManager()
        outline_tool = CourseOutlineTool(mock_vs)
        tool_manager.register_tool(outline_tool)

        result = tool_manager.execute_tool(
            "get_course_outline",
            course_name="NonExistent"
        )

        assert "No course found" in result
        assert "NonExistent" in result


class TestToolResultsToAI:
    """Tests for passing tool results back to Claude."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_results_included_in_follow_up_call(self, mock_anthropic_class):
        """Verify tool results are passed back to Claude for synthesis."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()

        # First response: Claude wants to use search tool
        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "machine learning"},
            tool_id="tool_abc"
        )

        # Second response: Final synthesized answer
        final_response = build_claude_text_response(
            "Based on the course materials, machine learning is..."
        )

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        # Setup real tool manager with mock VectorStore
        mock_vs = MagicMock()
        mock_vs.search.return_value = build_search_results(
            documents=["ML is a subset of AI."],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}]
        )
        mock_vs.get_lesson_link.return_value = "https://ml.com/1"

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        tools = [search_tool.get_tool_definition()]

        generator = AIGenerator(api_key="test", model="test")
        result = generator.generate_response(
            query="What is ML?",
            tools=tools,
            tool_manager=tool_manager
        )

        # Verify second API call includes tool result
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Find tool_result message
        tool_result_msg = None
        for msg in messages:
            if msg["role"] == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            tool_result_msg = item
                            break

        assert tool_result_msg is not None
        assert "ML Course" in tool_result_msg["content"]

        # Verify final result
        assert "machine learning" in result.lower()


class TestNoToolUsePath:
    """Tests for direct response path (no tool use)."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_direct_response_without_tool_use(self, mock_anthropic_class):
        """Verify direct response when Claude doesn't use tools."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Python is a programming language."
        )
        mock_anthropic_class.return_value = mock_client

        mock_vs = MagicMock()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        tools = [search_tool.get_tool_definition()]

        generator = AIGenerator(api_key="test", model="test")
        result = generator.generate_response(
            query="What is Python?",
            tools=tools,
            tool_manager=tool_manager
        )

        # Only one API call
        assert mock_client.messages.create.call_count == 1

        # VectorStore should not be called
        mock_vs.search.assert_not_called()

        # Direct response returned
        assert result == "Python is a programming language."

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_sources_for_direct_response(self, mock_anthropic_class):
        """Verify no sources tracked when no tool is used."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "General knowledge answer."
        )
        mock_anthropic_class.return_value = mock_client

        mock_vs = MagicMock()
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        tools = [search_tool.get_tool_definition()]

        generator = AIGenerator(api_key="test", model="test")
        generator.generate_response(
            query="What is 2+2?",
            tools=tools,
            tool_manager=tool_manager
        )

        # Sources should be empty
        sources = tool_manager.get_last_sources()
        assert sources == []


class TestSourcePopulation:
    """Tests for source population after searches."""

    def test_sources_populated_after_search(self):
        """Verify sources are correctly populated after search."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = build_search_results(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ]
        )
        mock_vs.get_lesson_link.side_effect = [
            "https://coursea.com/1",
            "https://courseb.com/2"
        ]

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        tool_manager.execute_tool("search_course_content", query="test")

        sources = tool_manager.get_last_sources()

        assert len(sources) == 2
        assert sources[0]["title"] == "Course A"
        assert sources[0]["lesson"] == 1
        assert sources[0]["url"] == "https://coursea.com/1"
        assert sources[1]["title"] == "Course B"
        assert sources[1]["lesson"] == 2
        assert sources[1]["url"] == "https://courseb.com/2"

    def test_sources_reset_clears_previous(self):
        """Verify reset_sources clears sources from previous queries."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = build_search_results(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}]
        )
        mock_vs.get_lesson_link.return_value = "https://example.com"

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        # First search
        tool_manager.execute_tool("search_course_content", query="first")
        assert len(tool_manager.get_last_sources()) == 1

        # Reset
        tool_manager.reset_sources()
        assert len(tool_manager.get_last_sources()) == 0

    def test_sources_empty_when_no_results(self):
        """Verify sources are empty when search returns no results."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = build_search_results(
            documents=[],
            metadata=[],
            distances=[]
        )

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        tool_manager.register_tool(search_tool)

        tool_manager.execute_tool("search_course_content", query="nothing")

        # Sources should be empty for empty results
        sources = tool_manager.get_last_sources()
        assert sources == []


class TestMultipleToolRegistration:
    """Tests for systems with multiple registered tools."""

    def test_both_tools_registered_and_accessible(self):
        """Verify both search and outline tools can be registered."""
        mock_vs = MagicMock()

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        outline_tool = CourseOutlineTool(mock_vs)

        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_correct_tool_executed_by_name(self):
        """Verify correct tool is executed based on name."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = build_search_results(
            documents=["Search result"],
            metadata=[{"course_title": "C", "lesson_number": 1}]
        )
        mock_vs.get_lesson_link.return_value = "http://test"
        mock_vs.get_course_outline.return_value = {
            "title": "Outline Course",
            "course_link": "http://outline",
            "lessons": []
        }

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vs)
        outline_tool = CourseOutlineTool(mock_vs)
        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)

        # Execute search
        search_result = tool_manager.execute_tool(
            "search_course_content",
            query="test"
        )
        assert "Search result" in search_result

        # Execute outline
        outline_result = tool_manager.execute_tool(
            "get_course_outline",
            course_name="test"
        )
        assert "Outline Course" in outline_result
