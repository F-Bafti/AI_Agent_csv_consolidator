import time
import traceback
from dataclasses import dataclass
from typing import List, Callable, Dict, Any

# 1- Goal Class
# Creates a simple container to hold information about what the agent should accomplish
# priority: How important this goal is (lower numbers = higher priority)
# name: Short name for the goal (like "Gather Information")
# description: Detailed explanation of what to do
# frozen=True: Makes this immutable - once created, it can't be changed
@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str



# 2- Action Class
# Creates a wrapper around a Python function to make it available to the AI
# name: What the AI will call this action (like "read_file")
# function: The actual Python function to execute
# description: Tells the AI what this function does
# parameters: Describes what arguments the function needs (JSON schema format)
# terminal: Whether calling this action should end the agent's execution
class Action:
    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 parameters: Dict,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        """Execute the action's function"""
        return self.function(**args)

# Creates a container to store all available actions
class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    # Looks up an action by its name
    def get_action(self, name: str) -> Action | None:
        return self.actions.get(name, None)

    # Returns ALL actions as a list
    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())



# 3- Memory Class
# Creates a container to store the conversation history
# self.items = []: An empty list to hold memory items
# Each item will be a dictionary representing one piece of the conversation
class Memory:
    def __init__(self):
        self.items = []  # Basic conversation history

    # Adds a new memory item to the end of the list
    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    # Returns the stored memories as a list
    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        return self.items[:limit]

    # Creates a new Memory object with system messages filtered out
    def copy_without_system_memories(self):
        """Return a copy of the memory without system memories"""
        filtered_items = [m for m in self.items if m["type"] != "system"]
        memory = Memory()
        memory.items = filtered_items
        return memory



# 4- Environment Class
# This is where actions actually get executed safely
# try: attempts to run the action
# action.execute(**args) calls the action with the provided arguments
# If it works: calls self.format_result() to package the result nicely
# If it fails: catches the error and returns error information instead of crashing
class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }