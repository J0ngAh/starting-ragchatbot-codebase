import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
- **search_course_content**: Search within course content for specific topics or information
- **get_course_outline**: Get course structure including title, link, and complete lesson list

Tool Usage Guidelines:
- Use **get_course_outline** for: course structure, lesson lists, course overview, "what lessons are in...", "outline of..."
- Use **search_course_content** for: specific topics, detailed content, particular concepts within courses
- **Sequential tool calls**: You may use up to 2 tool calls for complex queries:
  - First get a course outline to identify lesson topics
  - Then search for related content based on what you learned
- Synthesize results into accurate, fact-based responses
- If no results found, state this clearly

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **Multi-step questions**: Break down into sequential tool calls when needed
- **No meta-commentary**: Provide direct answers only
- Do not mention "based on the search results" or "according to the tool"
- **For course outlines**: Always include the course title, course link, and all lesson numbers with titles

All responses must be:
1. **Brief and concise** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when helpful
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls with support for sequential tool rounds.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters (includes tools)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        MAX_TOOL_ROUNDS = 2
        current_response = initial_response
        messages = base_params["messages"].copy()

        for _ in range(MAX_TOOL_ROUNDS):
            # Add assistant's tool_use response
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Keep tools in API params for potential follow-up calls
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params.get("tools"),
                "tool_choice": {"type": "auto"}
            }

            current_response = self.client.messages.create(**api_params)

            # Exit early if Claude doesn't need more tools
            if current_response.stop_reason != "tool_use":
                return current_response.content[0].text

        # Max rounds reached - extract text from final response
        for block in current_response.content:
            if hasattr(block, 'text'):
                return block.text
        return "Unable to generate response after tool execution."