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

🎯 CENTER-SPECIFIC FILE QUERIES (when user mentions a specific center name):
- "how many files from [center]" → count_center_csv_files(center_keyword="[center]", path="...")
- "count files from [center]" → count_center_csv_files(center_keyword="[center]", path="...")  
- "how many [center] files" → count_center_csv_files(center_keyword="[center]", path="...")
- "list files from [center]" → list_center_csv_files(center_keyword="[center]", path="...")
- "show [center] files" → list_center_csv_files(center_keyword="[center]", path="...")
- "files related to [center]" → list_center_csv_files(center_keyword="[center]", path="...")

🎯 ALL FILES QUERIES (when user wants to see everything):
- "list all files" → list_csv_files_in_dir(path="...")
- "show all csvs" → list_csv_files_in_dir(path="...")
- "what files are in [directory]" → list_csv_files_in_dir(path="[directory]")
- "list files in [directory]" → list_csv_files_in_dir(path="[directory]")

🎯 CURRENT DIRECTORY QUERIES:
- "list csv files" (while in target dir) → list_csv_files()
- "show csvs" (while in target dir) → list_csv_files()
- "how many csv files" (in current dir) → count_csv_files()
- "count csv files" (in current dir) → count_csv_files()

🎯 SMART PATH DETECTION QUERIES:
- "how many csv files in [directory]" → count_csv_files(path="[directory]")
- "count files in [directory]" → count_csv_files(path="[directory]")
- "how many csv files in input_csvs" → count_csv_files(path="input_csvs")

🎯 FILE CLEANING QUERIES:
- "clean [filename]" → clean_csv_file(file_path="[filename]")
- "clean all files" → clean_all_csv_files()
- "clean all csvs" → clean_all_csv_files()  
- "clean everything" → clean_all_csv_files()
- "process all files" → clean_all_csv_files()
- "batch clean files" → clean_all_csv_files()
- "clean all files in [directory]" → clean_all_csv_files(path="[directory]")
- "clean all files and show preview" → clean_all_csv_files_with_preview()
- "process all csvs with preview" → clean_all_csv_files_with_preview()

📝 EXAMPLES OF CORRECT TOOL SELECTION:
- "how many files from neyshabour" → count_center_csv_files(center_keyword="neyshabour")
- "how many files we have from neyshabour" → count_center_csv_files(center_keyword="neyshabour") 
- "list files from boushehr" → list_center_csv_files(center_keyword="boushehr")
- "show all files in input_csvs" → list_csv_files_in_dir(path="input_csvs")
- "list all the files in input_csvs" → list_csv_files_in_dir(path="input_csvs")

⚠️ COMMON MISTAKES TO AVOID:
- DO NOT use list_csv_files_in_dir for center-specific queries
- DO NOT use list_csv_files for queries that specify a directory path
- ALWAYS extract the center name from user queries like "neyshabour", "boushehr", "sanandaj"

🔍 CENTER NAME EXTRACTION:
- "neyshabour" → center_keyword="neyshabour"
- "boushehr" → center_keyword="boushehr"  
- "sanandaj" → center_keyword="sanandaj"
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

    def parse_response(self, response: str) -> dict:
        try:
            parsed = json.loads(response)
            if "args" not in parsed or not isinstance(parsed["args"], dict):
                parsed["args"] = {}
            return parsed
        except Exception:
            return {
                "tool": "terminate",
                "args": {"message": response}
            }


class PythonActionRegistry(ActionRegistry):
    def __init__(self, tags: List[str] = None, tool_names: List[str] = None):
        super().__init__()

        self.terminate_tool = None

        for tool_name, tool_desc in tools.items():
            if tool_name == "terminate":
                self.terminate_tool = tool_desc

            if tool_names and tool_name not in tool_names:
                continue

            tool_tags = tool_desc.get("tags", [])
            if tags and not any(tag in tool_tags for tag in tags):
                continue

            self.register(Action(
                name=tool_name,
                function=tool_desc["function"],
                description=tool_desc["description"],
                parameters=tool_desc.get("parameters", {}),
                terminal=tool_desc.get("terminal", False)
            ))

    def register_terminate_tool(self):
        if self.terminate_tool:
            self.register(Action(
                name="terminate",
                function=self.terminate_tool["function"],
                description=self.terminate_tool["description"],
                parameters=self.terminate_tool.get("parameters", {}),
                terminal=self.terminate_tool.get("terminal", False)
            ))
        else:
            raise Exception("Terminate tool not found in tool registry")