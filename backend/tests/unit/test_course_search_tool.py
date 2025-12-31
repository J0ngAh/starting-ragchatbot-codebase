"""
Unit tests for CourseSearchTool.execute() method.

Tests evaluate:
1. Formatted output for successful search results
2. Empty result handling
3. Error handling
4. Source tracking
5. Parameter passing to VectorStore
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add backend and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from search_tools import CourseSearchTool
from vector_store import SearchResults
from helpers import build_search_results, build_empty_search_results


class TestCourseSearchToolDefinition:
    """Tests for tool definition schema."""

    def test_tool_definition_has_correct_name(self, mock_vector_store):
        """Verify tool name is 'search_course_content'."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"

    def test_tool_definition_has_description(self, mock_vector_store):
        """Verify tool has a meaningful description."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert "description" in definition
        assert len(definition["description"]) > 0

    def test_tool_definition_has_valid_schema(self, mock_vector_store):
        """Verify input_schema is properly structured."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert "input_schema" in definition
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_tool_definition_requires_query(self, mock_vector_store):
        """Verify query is the only required parameter."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["input_schema"]["required"] == ["query"]

    def test_tool_definition_has_optional_course_name(self, mock_vector_store):
        """Verify course_name is an optional parameter."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        properties = definition["input_schema"]["properties"]
        assert "course_name" in properties
        assert "course_name" not in definition["input_schema"]["required"]

    def test_tool_definition_has_optional_lesson_number(self, mock_vector_store):
        """Verify lesson_number is an optional parameter."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        properties = definition["input_schema"]["properties"]
        assert "lesson_number" in properties
        assert properties["lesson_number"]["type"] == "integer"


class TestCourseSearchToolExecuteSuccess:
    """Tests for successful search result formatting."""

    def test_execute_returns_formatted_results_single_doc(self, mock_vector_store):
        """Verify single result is formatted with header."""
        # Setup single result
        mock_vector_store.search.return_value = build_search_results(
            documents=["Content about neural networks."],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}],
            distances=[0.3]
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="neural networks")

        assert "[ML Course - Lesson 1]" in result
        assert "Content about neural networks." in result

    def test_execute_returns_formatted_results_multiple_docs(self, mock_vector_store):
        """Verify multiple results are separated by double newline."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["First result content.", "Second result content."],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.3, 0.5]
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test query")

        # Results should be separated by \n\n
        assert "\n\n" in result
        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result

    def test_execute_includes_course_title_in_header(self, mock_vector_store):
        """Verify course title appears in header."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["Some content."],
            metadata=[{"course_title": "Advanced Python", "lesson_number": None}]
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="python")

        assert "[Advanced Python]" in result

    def test_execute_includes_lesson_number_when_present(self, mock_vector_store):
        """Verify lesson number appears in header when available."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["Lesson content."],
            metadata=[{"course_title": "ML Basics", "lesson_number": 3}]
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="basics")

        assert "Lesson 3" in result
        assert "[ML Basics - Lesson 3]" in result

    def test_execute_omits_lesson_from_header_when_none(self, mock_vector_store):
        """Verify header format when lesson_number is None."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["Content without lesson."],
            metadata=[{"course_title": "General Course", "lesson_number": None}]
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="general")

        assert "[General Course]" in result
        assert "Lesson" not in result


class TestCourseSearchToolExecuteEmpty:
    """Tests for empty result handling."""

    def test_execute_returns_no_content_message_when_empty(self, mock_vector_store_empty):
        """Verify empty results return appropriate message."""
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_includes_course_filter_in_message(self, mock_vector_store_empty):
        """Verify course filter mentioned in empty result message."""
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute(query="test", course_name="Python Basics")

        assert "course 'Python Basics'" in result

    def test_execute_empty_includes_lesson_filter_in_message(self, mock_vector_store_empty):
        """Verify lesson filter mentioned in empty result message."""
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute(query="test", lesson_number=5)

        assert "lesson 5" in result

    def test_execute_empty_includes_both_filters_in_message(self, mock_vector_store_empty):
        """Verify both filters mentioned when both provided."""
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute(query="test", course_name="ML", lesson_number=2)

        assert "course 'ML'" in result
        assert "lesson 2" in result


class TestCourseSearchToolExecuteError:
    """Tests for error handling."""

    def test_execute_returns_error_when_search_fails(self, mock_vector_store_error):
        """Verify error message is returned when search fails."""
        tool = CourseSearchTool(mock_vector_store_error)
        result = tool.execute(query="test")

        assert "Search error" in result

    def test_execute_returns_course_not_found_error(self, mock_vector_store):
        """Verify course not found error is passed through."""
        mock_vector_store.search.return_value = SearchResults.empty(
            "No course found matching 'NonExistent Course'"
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="NonExistent Course")

        assert "No course found matching" in result


class TestCourseSearchToolParameterPassing:
    """Tests for verifying parameters are passed correctly to VectorStore."""

    def test_execute_passes_query_to_vector_store(self, mock_vector_store):
        """Verify query parameter is passed to search."""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="machine learning")

        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["query"] == "machine learning"

    def test_execute_passes_course_name_to_vector_store(self, mock_vector_store):
        """Verify course_name parameter is passed to search."""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", course_name="Python Course")

        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["course_name"] == "Python Course"

    def test_execute_passes_lesson_number_to_vector_store(self, mock_vector_store):
        """Verify lesson_number parameter is passed to search."""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", lesson_number=3)

        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["lesson_number"] == 3

    def test_execute_handles_none_optional_parameters(self, mock_vector_store):
        """Verify None values for optional params are handled."""
        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", course_name=None, lesson_number=None)

        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["course_name"] is None
        assert call_kwargs["lesson_number"] is None


class TestCourseSearchToolSourceTracking:
    """Tests for source tracking functionality."""

    def test_execute_tracks_sources_correctly(self, mock_vector_store):
        """Verify last_sources is populated with correct structure."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["Content here."],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        assert source["title"] == "Test Course"
        assert source["lesson"] == 1
        assert source["url"] == "https://example.com/lesson/1"

    def test_execute_calls_get_lesson_link_for_each_result(self, mock_vector_store):
        """Verify URL lookup is called for each result with lesson number."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["Doc 1", "Doc 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ]
        )

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        # Should be called twice, once for each result
        assert mock_vector_store.get_lesson_link.call_count == 2

    def test_execute_does_not_call_get_lesson_link_when_lesson_is_none(self, mock_vector_store):
        """Verify URL lookup is skipped when lesson_number is None."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["Doc 1"],
            metadata=[{"course_title": "Course A", "lesson_number": None}]
        )

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        # Should not be called when lesson_number is None
        mock_vector_store.get_lesson_link.assert_not_called()

    def test_last_sources_reset_on_new_execute(self, mock_vector_store):
        """Verify sources are replaced on each execute call."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["First doc"],
            metadata=[{"course_title": "First Course", "lesson_number": 1}]
        )

        tool = CourseSearchTool(mock_vector_store)

        # First execute
        tool.execute(query="first")
        assert tool.last_sources[0]["title"] == "First Course"

        # Change mock response
        mock_vector_store.search.return_value = build_search_results(
            documents=["Second doc"],
            metadata=[{"course_title": "Second Course", "lesson_number": 2}]
        )

        # Second execute should replace sources
        tool.execute(query="second")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["title"] == "Second Course"

    def test_last_sources_empty_on_empty_results(self, mock_vector_store_empty):
        """Verify sources are empty when no results found."""
        tool = CourseSearchTool(mock_vector_store_empty)
        tool.execute(query="nothing")

        # Sources should not be set for empty results (execute returns early)
        # The last_sources is initialized as empty list
        assert tool.last_sources == []

    def test_source_structure_has_required_keys(self, mock_vector_store):
        """Verify each source has title, lesson, and url keys."""
        mock_vector_store.search.return_value = build_search_results(
            documents=["Content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}]
        )

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        source = tool.last_sources[0]
        assert "title" in source
        assert "lesson" in source
        assert "url" in source
