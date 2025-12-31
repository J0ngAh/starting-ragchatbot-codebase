"""
Unit tests for ToolManager class.

Tests evaluate:
1. Tool registration
2. Tool definition retrieval
3. Tool execution dispatch
4. Source tracking and reset
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from search_tools import ToolManager, Tool


class MockTool(Tool):
    """Mock tool for testing ToolManager."""

    def __init__(self, name: str, has_sources: bool = False):
        self._name = name
        self._has_sources = has_sources
        if has_sources:
            self.last_sources = []

    def get_tool_definition(self):
        return {
            "name": self._name,
            "description": f"Mock tool: {self._name}",
            "input_schema": {"type": "object", "properties": {}}
        }

    def execute(self, **kwargs):
        if self._has_sources:
            self.last_sources = [{"source": "test"}]
        return f"Executed {self._name} with {kwargs}"


class TestToolManagerRegistration:
    """Tests for tool registration."""

    def test_register_tool_adds_to_tools_dict(self):
        """Verify tool is added to internal tools dictionary."""
        manager = ToolManager()
        tool = MockTool("test_tool")

        manager.register_tool(tool)

        assert "test_tool" in manager.tools

    def test_register_tool_uses_tool_name_as_key(self):
        """Verify tool name from definition is used as key."""
        manager = ToolManager()
        tool = MockTool("my_custom_tool")

        manager.register_tool(tool)

        assert manager.tools["my_custom_tool"] == tool

    def test_register_multiple_tools(self):
        """Verify multiple tools can be registered."""
        manager = ToolManager()
        tool1 = MockTool("tool_one")
        tool2 = MockTool("tool_two")

        manager.register_tool(tool1)
        manager.register_tool(tool2)

        assert len(manager.tools) == 2
        assert "tool_one" in manager.tools
        assert "tool_two" in manager.tools

    def test_register_tool_raises_error_for_missing_name(self):
        """Verify error when tool definition has no name."""
        manager = ToolManager()

        # Create tool with broken definition
        bad_tool = MagicMock()
        bad_tool.get_tool_definition.return_value = {"description": "No name"}

        with pytest.raises(ValueError, match="must have a 'name'"):
            manager.register_tool(bad_tool)


class TestToolManagerDefinitions:
    """Tests for tool definition retrieval."""

    def test_get_tool_definitions_returns_all_definitions(self):
        """Verify all tool definitions are returned."""
        manager = ToolManager()
        manager.register_tool(MockTool("tool_a"))
        manager.register_tool(MockTool("tool_b"))

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_get_tool_definitions_empty_when_no_tools(self):
        """Verify empty list when no tools registered."""
        manager = ToolManager()

        definitions = manager.get_tool_definitions()

        assert definitions == []


class TestToolManagerExecution:
    """Tests for tool execution dispatch."""

    def test_execute_tool_calls_correct_tool(self):
        """Verify correct tool is called by name."""
        manager = ToolManager()
        tool = MockTool("search_tool")
        manager.register_tool(tool)

        result = manager.execute_tool("search_tool", query="test")

        assert "Executed search_tool" in result

    def test_execute_tool_passes_kwargs(self):
        """Verify kwargs are passed to tool."""
        manager = ToolManager()
        tool = MockTool("my_tool")
        manager.register_tool(tool)

        result = manager.execute_tool("my_tool", param1="value1", param2="value2")

        assert "param1" in result
        assert "value1" in result

    def test_execute_tool_returns_not_found_for_unknown(self):
        """Verify error message for unknown tool."""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_execute_tool_returns_string(self):
        """Verify execute returns string result."""
        manager = ToolManager()
        manager.register_tool(MockTool("test"))

        result = manager.execute_tool("test")

        assert isinstance(result, str)


class TestToolManagerSourceTracking:
    """Tests for source tracking functionality."""

    def test_get_last_sources_returns_from_tool_with_sources(self):
        """Verify sources are retrieved from tool with last_sources."""
        manager = ToolManager()
        tool = MockTool("search_tool", has_sources=True)
        manager.register_tool(tool)

        # Execute to populate sources
        manager.execute_tool("search_tool")

        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]["source"] == "test"

    def test_get_last_sources_returns_empty_when_no_sources(self):
        """Verify empty list when no tool has sources."""
        manager = ToolManager()
        tool = MockTool("no_source_tool", has_sources=False)
        manager.register_tool(tool)

        sources = manager.get_last_sources()

        assert sources == []

    def test_get_last_sources_finds_first_tool_with_sources(self):
        """Verify first tool with non-empty sources is used."""
        manager = ToolManager()
        tool1 = MockTool("tool_1", has_sources=True)
        tool2 = MockTool("tool_2", has_sources=True)
        manager.register_tool(tool1)
        manager.register_tool(tool2)

        # Only execute first tool
        manager.execute_tool("tool_1")

        sources = manager.get_last_sources()

        # Should find sources from tool_1
        assert len(sources) == 1

    def test_reset_sources_clears_all_tool_sources(self):
        """Verify reset clears sources from all tools."""
        manager = ToolManager()
        tool = MockTool("search", has_sources=True)
        manager.register_tool(tool)

        # Execute to populate sources
        manager.execute_tool("search")
        assert len(manager.get_last_sources()) == 1

        # Reset
        manager.reset_sources()

        # Sources should be empty now
        assert manager.get_last_sources() == []

    def test_reset_sources_handles_tools_without_sources_attr(self):
        """Verify reset doesn't fail for tools without last_sources."""
        manager = ToolManager()
        tool = MockTool("no_sources", has_sources=False)
        manager.register_tool(tool)

        # Should not raise
        manager.reset_sources()

        assert manager.get_last_sources() == []
