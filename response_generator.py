from dataclasses import dataclass, field
import json
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Dict

@dataclass
class Prompt:
    messages: List[Dict] = field(default_factory=list)
    tools: List[Dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Fixing mutable default issue



def generate_response(prompt: Prompt) -> str:
    """Call LLM to get response"""
    # print("DEBUG: Starting generate_response")
    # print(f"DEBUG: Prompt has {len(prompt.messages)} messages and {len(prompt.tools)} tools")

    try:
        # Initialize Cohere model with LangChain
        llm = ChatCohere(
            model="command-r-plus",
            max_tokens=1024,
            temperature=0.3,
            timeout=30
        )
        # print("DEBUG: Cohere model initialized")

        messages = prompt.messages
        tools = prompt.tools
        result = None

        # Convert dict messages to LangChain message objects
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        # print(f"DEBUG: Converted {len(langchain_messages)} messages")

        if not tools:
            # print("DEBUG: No tools, making simple invoke call")
            response = llm.invoke(langchain_messages)
            result = response.content
        else:
            # print("DEBUG: Tools present, setting up tool calling")

            formatted_tools = []
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    func_def = tool["function"]
                    cohere_tool = {
                        "title": func_def["name"],
                        "description": func_def["description"],
                        "properties": func_def["parameters"].get("properties", {}),
                        "required": func_def["parameters"].get("required", [])
                    }
                    formatted_tools.append(cohere_tool)

            llm_with_tools = llm.bind_tools(formatted_tools)
            # print("DEBUG: About to invoke LLM with tools (this may take 30-60 seconds)")

            def make_call():
                return llm_with_tools.invoke(langchain_messages)

            with ThreadPoolExecutor() as executor:
                future = executor.submit(make_call)
                try:
                    response = future.result(timeout=60)
                    # print("DEBUG: Got response from LLM with tools")
                    if hasattr(response, "tool_calls"):
                        response.tool_calls = [
                            {
                                "name": tc.get("name"),
                                "args": tc.get("args") or {}
                            }
                            for tc in response.tool_calls
                        ]

                except FuturesTimeoutError:
                    # print("DEBUG: LLM call timed out after 60 seconds")
                    return "ERROR: LLM call timed out"

            # 🔐 Sanitize tool calls to avoid None in args
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                args = tool_call.get("args") or {}  # default to empty dict if None
                result = {
                    "tool": tool_call["name"],
                    "args": args,
                }
                result = json.dumps(result)
            else:
                result = response.content

        print(f"DEBUG: Returning result: {result}")
        return result

    except Exception as e:
        # print(f"DEBUG: Exception occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {str(e)}"
