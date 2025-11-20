import json
from typing import List, Any
from response_generator import Prompt
from tool_registry import tools
from GAME import Goal, Action, ActionRegistry, Memory, Environment

# What this class do?
# Example: user_input = "Write a README for this project."
# What construct_prompt Does:
# The agent takes that simple user input and builds a complex prompt that includes:
# System instructions (the goals), Conversation history, Available tools, The user's request
# Prompt(
#     messages=[
#         {"role": "system", "content": "Goal: Read each file..."},
#         {"role": "user", "content": "Write a README for this project."},
#         {"role": "assistant", "content": '{"tool": "list_files", "args": {}}'},
#         {"role": "user", "content": "Tool result: [file1.py, file2.py]"}
#     ],
#     tools=[list_files_tool, read_file_tool, terminate_tool]
# )

class AgentLanguage:
    def __init__(self):
        pass

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")

    def parse_response(self, response: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")



class AgentFunctionCallingActionLanguage(AgentLanguage):

    def __init__(self):
        super().__init__()

    def format_goals(self, goals: List[Goal]) -> List:
        # Map all goals to a single string that concatenates their description
        # and combine into a single message of type system
        sep = "\n-------------------\n"
        goal_instructions = "\n\n".join([f"{goal.name}:{sep}{goal.description}{sep}" for goal in goals])
        
        # Add comprehensive tool selection guidance
        tool_selection_guide = """
CRITICAL TOOL SELECTION GUIDELINES:
=================================

🎯 CENTER-SPECIFIC FILE QUERIES:
- "how many files from [center]" → count_center_csv_files(center_keyword="[center]", path="...")
- "count files from [center]" → count_center_csv_files(center_keyword="[center]", path="...")  
- "list files from [center]" → list_center_csv_files(center_keyword="[center]", path="...")
- "show [center] files" → list_center_csv_files(center_keyword="[center]", path="...")

🎯 ALL FILES QUERIES (General Directory Listing):
- "list all files" → list_directory_contents(path=".")
- "show contents" → list_directory_contents(path=".")
- "ls -lrt" → list_directory_contents(path=".")
- "what files are in [directory]" → list_directory_contents(path="[directory]")
- "list files in [directory]" → list_directory_contents(path="[directory]")

🎯 CSV FILES QUERIES (Simple Listing and Counting):
- "list all csvs" → list_csv_files_in_dir(path=".")
- "show all csvs" → list_csv_files_in_dir(path=".")
- "what csvs are here" → list_csv_files_in_dir(path=".")
- "how many csv files" → count_csv_files(path=".")
- "count csv files" → count_csv_files(path=".")
- "how many csv files in [directory]" → count_csv_files(path="[directory]")

🎯 CSV DETAIL QUERIES:
- "ls -lrt for csvs" → list_detailed_csv_files(path=".")
- "list detailed csv files" → list_detailed_csv_files(path=".")
- "show size and date of csvs" → list_detailed_csv_files(path=".")

🎯 FILE CLEANING QUERIES:
- "clean [filename]" → clean_csv_file(file_path="[filename]")
- "clean all csvs" → clean_all_csv_files(path=".")
- "process all files" → clean_all_csv_files(path=".")
- "batch clean files" → clean_all_csv_files(path=".")
- "clean all files in [directory]" → clean_all_csv_files(path="[directory]")
- "clean all files and show preview" → clean_all_csv_files_with_preview()

🎯 TERMINATION QUERIES:
- "bye" → terminate(message="Goodbye! Task complete.")
- "exit" → terminate(message="Task complete. Terminating.")
- "I'm done" / "That's all" / "thanks" → terminate(message="Task complete. Terminating.")
- If the conversation is clearly finished and the user has expressed satisfaction or intent to end.
=================================
"""
        
        return [
            {"role": "system", "content": goal_instructions + "\n\n" + tool_selection_guide}
        ]

    def format_memory(self, memory: Memory) -> List:
        """Generate response from language model"""
        items = memory.get_memories()
        mapped_items = []
        for item in items:

            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=4)

            if item["type"] == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            elif item["type"] == "environment":
                # Map environment results to user messages for Cohere compatibility
                mapped_items.append({"role": "user", "content": f"Tool result: {content}"})
            else:
                mapped_items.append({"role": "user", "content": content})

        return mapped_items

    def format_actions(self, actions: List[Action]) -> List:
        """Convert actions to LangChain-compatible tool format"""

        tools = []
        for action in actions:
            # Convert to OpenAI function format that LangChain can use
            tool_def = {
                "type": "function",
                "function": {
                    "name": action.name,
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                }
            }
            tools.append(tool_def)

        return tools

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:

        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)

        tools = self.format_actions(actions)

        return Prompt(messages=prompt, tools=tools)

    def adapt_prompt_after_parsing_error(self,
                                         prompt: Prompt,
                                         response: str,
                                         traceback: str,
                                         error: Any,
                                         retries_left: int) -> Prompt:

        return prompt

    # In language.py, update parse_response

    def parse_response(self, response: str) -> dict:
        try:
            # 1. First, try to parse the LLM's response as a tool call (JSON)
            parsed = json.loads(response)
            if "args" not in parsed or not isinstance(parsed["args"], dict):
                parsed["args"] = {}
            return parsed
        except Exception:
            # 2. If JSON parsing fails (i.e., the LLM returned plain text),
            #    force the agent to call the 'say' tool with the plain text.
            return {
                "tool": "say",  # <--- CHANGED FROM 'terminate' TO 'say'
                "args": {"message": response}
            }


class PythonActionRegistry(ActionRegistry):
    def __init__(self, tags: List[str] = None, tool_names: List[str] = None):
        super().__init__()

        self.terminate_tool = None
        
        # Collect tools based on tags/names
        tools_to_register = []
        for tool_name, tool_desc in tools.items():
            if tool_name == "terminate":
                self.terminate_tool = tool_desc
                continue # Skip registering here, handle separately

            tool_tags = tool_desc.get("tags", [])
            if tags and not any(tag in tool_tags for tag in tags):
                continue

            tools_to_register.append((tool_name, tool_desc))

        # Register filtered tools
        for tool_name, tool_desc in tools_to_register:
            self.register(Action(
                name=tool_name,
                function=tool_desc["function"],
                description=tool_desc["description"],
                parameters=tool_desc.get("parameters", {}),
                terminal=tool_desc.get("terminal", False)
            ))
            
        # 🔑 FIX: Ensure terminate tool is registered last
        self.register_terminate_tool()

    def register_terminate_tool(self):
        if self.terminate_tool:
            self.register(Action(
                name="terminate",
                function=self.terminate_tool["function"],
                description=self.terminate_tool["description"],
                parameters=self.terminate_tool.get("parameters", {}),
                terminal=True # Force terminal to be True
            ))
        else:
            # You should check that system_tools.py successfully loaded
            raise Exception("Terminate tool not found in tool registry. Check system_tools.py import.")