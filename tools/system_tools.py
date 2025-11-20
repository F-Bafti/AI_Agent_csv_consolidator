from tool_registry import  register_tool

@register_tool(tags=["system"], terminal=True)
def terminate(message: str) -> str:
    """Terminates the agent's execution with a final message.

    Args:
        message: The final message to return before terminating

    Returns:
        The message with a termination note appended
    """
    return f"{message}\nTerminating..."


@register_tool(tags=["system"], terminal=False) # Important: terminal=False
def say(message: str) -> str:
    """
    USAGE: Use when you need to respond to the user with a purely conversational message,
    when no specific tool is required, or when the tool execution fails.
    
    Returns:
        The message to be displayed to the user.
    """
    return message

# Note: The existing 'terminate' tool is below this and remains terminal=True.
@register_tool(tags=["system"], terminal=True)
def terminate(message: str) -> str:
    # ... (remains unchanged)
    return f"{message}\nTerminating..."