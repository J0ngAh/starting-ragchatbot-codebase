"""
Unit tests for AIGenerator tool calling behavior.

Tests evaluate:
1. Direct response handling (no tool use)
2. Tool use detection and execution
3. System prompt and history handling
4. Tool result passing back to Claude
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add backend and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_generator import AIGenerator
from helpers import (
    build_claude_mixed_response,
    build_claude_text_response,
    build_claude_tool_use_response,
)


class TestAIGeneratorInitialization:
    """Tests for AIGenerator initialization."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_init_creates_anthropic_client(self, mock_anthropic_class):
        """Verify Anthropic client is created with API key."""
        generator = AIGenerator(api_key="test-key", model="claude-test")

        mock_anthropic_class.assert_called_once_with(api_key="test-key")

    @patch("ai_generator.anthropic.Anthropic")
    def test_init_sets_model_from_parameter(self, mock_anthropic_class):
        """Verify model is stored correctly."""
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

        assert generator.model == "claude-sonnet-4-20250514"

    @patch("ai_generator.anthropic.Anthropic")
    def test_base_params_set_correctly(self, mock_anthropic_class):
        """Verify base API parameters are configured."""
        generator = AIGenerator(api_key="test-key", model="claude-test")

        assert generator.base_params["model"] == "claude-test"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800


class TestAIGeneratorDirectResponse:
    """Tests for direct response handling (no tool use)."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_returns_text_for_direct_answer(
        self, mock_anthropic_class
    ):
        """Verify direct text response is extracted correctly."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Here is your answer."
        )
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(query="What is Python?")

        assert result == "Here is your answer."

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_query_in_messages(self, mock_anthropic_class):
        """Verify query is sent to API in messages."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Response"
        )
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(query="What is ML?")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is ML?"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_system_prompt(self, mock_anthropic_class):
        """Verify system prompt is included in API call."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Response"
        )
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(query="Test")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert "AI assistant" in call_kwargs["system"]

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_history_when_provided(
        self, mock_anthropic_class
    ):
        """Verify conversation history is appended to system prompt."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Response"
        )
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(
            query="Follow up question",
            conversation_history="User: Hello\nAssistant: Hi there!",
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system_content = call_kwargs["system"]

        assert "Previous conversation:" in system_content
        assert "User: Hello" in system_content
        assert "Assistant: Hi there!" in system_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_no_history_in_system_when_none(
        self, mock_anthropic_class
    ):
        """Verify clean system prompt when no history."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Response"
        )
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(query="Test", conversation_history=None)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system_content = call_kwargs["system"]

        assert "Previous conversation:" not in system_content


class TestAIGeneratorToolHandling:
    """Tests for tool definition handling."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_tools_when_provided(self, mock_anthropic_class):
        """Verify tools are included in API call when provided."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Response"
        )
        mock_anthropic_class.return_value = mock_client

        tools = [
            {"name": "test_tool", "description": "A test tool", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(query="Test", tools=tools)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_sets_tool_choice_auto(self, mock_anthropic_class):
        """Verify tool_choice is set to auto when tools provided."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Response"
        )
        mock_anthropic_class.return_value = mock_client

        tools = [{"name": "test_tool", "description": "Test", "input_schema": {}}]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(query="Test", tools=tools)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_no_tools_param_when_none(self, mock_anthropic_class):
        """Verify tools parameter not included when None."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Response"
        )
        mock_anthropic_class.return_value = mock_client

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(query="Test", tools=None)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs


class TestAIGeneratorToolExecution:
    """Tests for tool use detection and execution."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_detects_tool_use_stop_reason(self, mock_anthropic_class):
        """Verify tool_use stop_reason triggers tool execution."""
        mock_client = MagicMock()

        # First response triggers tool use
        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_123",
        )

        # Second response is final text
        final_response = build_claude_text_response("Final answer")

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        # Create mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        tools = [
            {"name": "search_course_content", "description": "Test", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Should have called API twice
        assert mock_client.messages.create.call_count == 2
        assert result == "Final answer"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_calls_tool_manager_execute(self, mock_anthropic_class):
        """Verify tool_manager.execute_tool is called."""
        mock_client = MagicMock()

        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "neural networks"},
            tool_id="tool_abc",
        )
        final_response = build_claude_text_response("Answer about neural networks")

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = (
            "[ML Course]\nNeural network content"
        )

        tools = [
            {"name": "search_course_content", "description": "Test", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(
            query="Tell me about neural networks",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once()

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_passes_correct_tool_input(self, mock_anthropic_class):
        """Verify tool input kwargs are forwarded correctly."""
        mock_client = MagicMock()

        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "python basics", "course_name": "Python 101"},
            tool_id="tool_xyz",
        )
        final_response = build_claude_text_response("Here's info about Python")

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Python content"

        tools = [
            {"name": "search_course_content", "description": "Test", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(
            query="Python basics", tools=tools, tool_manager=mock_tool_manager
        )

        mock_tool_manager.execute_tool.assert_called_with(
            "search_course_content", query="python basics", course_name="Python 101"
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution_adds_tool_results_message(
        self, mock_anthropic_class
    ):
        """Verify tool results are added to messages for follow-up."""
        mock_client = MagicMock()

        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_result_id",
        )
        final_response = build_claude_text_response("Based on the results...")

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        tools = [
            {"name": "search_course_content", "description": "Test", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Check second API call includes tool results
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        # Should have: user query, assistant tool_use, user tool_result
        assert len(messages) == 3

        # Last message should be tool result
        tool_result_msg = messages[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tool_result_id"
        assert tool_result_msg["content"][0]["content"] == "Tool execution result"

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution_makes_follow_up_call(self, mock_anthropic_class):
        """Verify second API call is made after tool execution."""
        mock_client = MagicMock()

        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_id",
        )
        final_response = build_claude_text_response("Final synthesized answer")

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        tools = [
            {"name": "search_course_content", "description": "Test", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution_returns_final_text(self, mock_anthropic_class):
        """Verify final text response is returned after tool execution."""
        mock_client = MagicMock()

        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_id",
        )
        final_response = build_claude_text_response(
            "This is the synthesized final answer"
        )

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        tools = [
            {"name": "search_course_content", "description": "Test", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        assert result == "This is the synthesized final answer"


class TestAIGeneratorMultipleToolCalls:
    """Tests for handling multiple tool calls in single response."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_handles_multiple_tool_calls(self, mock_anthropic_class):
        """Verify all tool calls are executed when multiple present."""
        mock_client = MagicMock()

        # Create response with two tool calls
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.input = {"query": "first query"}
        tool_block_1.id = "tool_1"

        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.input = {"course_name": "ML Course"}
        tool_block_2.id = "tool_2"

        multi_tool_response = Mock()
        multi_tool_response.stop_reason = "tool_use"
        multi_tool_response.content = [tool_block_1, tool_block_2]

        final_response = build_claude_text_response("Combined answer")

        mock_client.messages.create.side_effect = [multi_tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        tools = [
            {
                "name": "search_course_content",
                "description": "Test",
                "input_schema": {},
            },
            {"name": "get_course_outline", "description": "Test", "input_schema": {}},
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Both tools should have been executed
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result == "Combined answer"


class TestAIGeneratorNoToolManager:
    """Tests for behavior when tool_manager is not provided."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_returns_direct_response_without_tool_manager(self, mock_anthropic_class):
        """When tool_manager is None, tool_use response should be handled differently."""
        mock_client = MagicMock()

        # Even if Claude wants to use tools, without tool_manager we can't execute
        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_id",
        )
        # Add a text block for fallback
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "I would search for..."
        tool_response.content.insert(0, text_block)

        mock_client.messages.create.return_value = tool_response
        mock_anthropic_class.return_value = mock_client

        tools = [
            {"name": "search_course_content", "description": "Test", "input_schema": {}}
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")

        # Without tool_manager, should not crash and return something
        # Based on the code, it checks stop_reason == "tool_use" AND tool_manager
        # If tool_manager is None, it falls through to return response.content[0].text
        result = generator.generate_response(
            query="Test", tools=tools, tool_manager=None
        )

        # Should return the first content block's text
        assert result == "I would search for..."


class TestAIGeneratorSequentialToolCalls:
    """Tests for sequential tool calling behavior (up to 2 rounds)."""

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_sequential_tool_calls(self, mock_anthropic_class):
        """Verify 2 tools execute in sequence (3 API calls total)."""
        mock_client = MagicMock()

        # Round 1: get_course_outline
        outline_response = build_claude_tool_use_response(
            tool_name="get_course_outline",
            tool_input={"course_name": "ML Course"},
            tool_id="tool_1",
        )

        # Round 2: search_course_content
        search_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "neural networks"},
            tool_id="tool_2",
        )

        # Final: text response
        final_response = build_claude_text_response(
            "Found matching content in both courses."
        )

        mock_client.messages.create.side_effect = [
            outline_response,
            search_response,
            final_response,
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course: ML\nLesson 4: Neural Networks",
            "[Deep Learning]\nNeural networks content",
        ]

        tools = [
            {
                "name": "get_course_outline",
                "description": "Get outline",
                "input_schema": {},
            },
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            },
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(
            query="Find similar content to lesson 4 of ML Course",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify 3 API calls total
        assert mock_client.messages.create.call_count == 3

        # Verify both tools executed
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify final result
        assert result == "Found matching content in both courses."

    @patch("ai_generator.anthropic.Anthropic")
    def test_single_tool_call_still_works(self, mock_anthropic_class):
        """Backward compatibility - 1 tool works (2 API calls)."""
        mock_client = MagicMock()

        tool_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_id="tool_1",
        )

        # After first tool, Claude returns text (no second tool)
        final_response = build_claude_text_response("Here is the answer.")

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Should only have 2 API calls (initial + after tool)
        assert mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1
        assert result == "Here is the answer."

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_enforced(self, mock_anthropic_class):
        """Loop stops after 2 rounds even if Claude wants more tools."""
        mock_client = MagicMock()

        # Claude keeps requesting tools
        tool_response_1 = build_claude_tool_use_response(
            tool_name="get_course_outline",
            tool_input={"course_name": "A"},
            tool_id="tool_1",
        )
        tool_response_2 = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "B"},
            tool_id="tool_2",
        )
        # Third would-be tool call with text
        tool_response_3 = build_claude_mixed_response(
            text="Partial answer",
            tool_name="search_course_content",
            tool_input={"query": "C"},
            tool_id="tool_3",
        )

        mock_client.messages.create.side_effect = [
            tool_response_1,
            tool_response_2,
            tool_response_3,
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        tools = [
            {
                "name": "get_course_outline",
                "description": "Get outline",
                "input_schema": {},
            },
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            },
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Should have exactly 3 API calls (initial + 2 rounds)
        assert mock_client.messages.create.call_count == 3

        # Only 2 tool executions (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2

        # Should return partial text from mixed response
        assert result == "Partial answer"

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_available_in_follow_up_call(self, mock_anthropic_class):
        """Verify tools key present in 2nd API call."""
        mock_client = MagicMock()

        tool_response = build_claude_tool_use_response(
            tool_name="get_course_outline",
            tool_input={"course_name": "Test"},
            tool_id="tool_1",
        )
        final_response = build_claude_text_response("Done")

        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Outline"

        tools = [
            {
                "name": "get_course_outline",
                "description": "Get outline",
                "input_schema": {},
            },
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            },
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(
            query="Test", tools=tools, tool_manager=mock_tool_manager
        )

        # Check second API call includes tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tools"] == tools

    @patch("ai_generator.anthropic.Anthropic")
    def test_message_history_accumulates(self, mock_anthropic_class):
        """Verify messages contain full tool call history."""
        mock_client = MagicMock()

        outline_response = build_claude_tool_use_response(
            tool_name="get_course_outline",
            tool_input={"course_name": "ML"},
            tool_id="tool_outline",
        )
        search_response = build_claude_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "neural nets"},
            tool_id="tool_search",
        )
        final_response = build_claude_text_response("Final answer")

        mock_client.messages.create.side_effect = [
            outline_response,
            search_response,
            final_response,
        ]
        mock_anthropic_class.return_value = mock_client

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Outline result", "Search result"]

        tools = [
            {
                "name": "get_course_outline",
                "description": "Get outline",
                "input_schema": {},
            },
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            },
        ]

        generator = AIGenerator(api_key="test-key", model="claude-test")
        generator.generate_response(
            query="Find similar content", tools=tools, tool_manager=mock_tool_manager
        )

        # Check third (final) API call has full history
        third_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        messages = third_call_kwargs["messages"]

        # Expected: user query, assistant(tool1), user(result1), assistant(tool2), user(result2)
        assert len(messages) == 5

        # Verify structure
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
        assert messages[4]["content"][0]["type"] == "tool_result"

    @patch("ai_generator.anthropic.Anthropic")
    def test_no_tool_call_returns_direct_response(self, mock_anthropic_class):
        """No change to direct response behavior."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = build_claude_text_response(
            "Direct answer without tools."
        )
        mock_anthropic_class.return_value = mock_client

        tools = [
            {
                "name": "search_course_content",
                "description": "Search",
                "input_schema": {},
            }
        ]
        mock_tool_manager = MagicMock()

        generator = AIGenerator(api_key="test-key", model="claude-test")
        result = generator.generate_response(
            query="What is Python?", tools=tools, tool_manager=mock_tool_manager
        )

        # Only 1 API call - no tool execution
        assert mock_client.messages.create.call_count == 1
        assert mock_tool_manager.execute_tool.call_count == 0
        assert result == "Direct answer without tools."
