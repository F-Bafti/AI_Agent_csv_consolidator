In this tutorial (https://lnkd.in/ehJGx_YB), I walk through how I developed an AI agent that can process, clean, and consolidate CSV files automatically. This project combines concepts I learned from Coursera's "AI Agents and Agentic AI with Python & Generative AI" courses.
What it does: 
- Navigate to specific directories
- List and identify CSV files
- Read and analyze file contents
- Clean up messy data
- Consolidate multiple files into one
You just chat with it naturally! This is the link to Github repo for the agent.

## Summary:

1. GAME.py
	•	Core classes: Goal, Action, ActionRegistry, Memory, Environment.
	•	Defines the structure for goals, actions, storing memory, and executing actions safely.

2. agent.py
	•	Wraps everything into an Agent.
	•	Loops through:
	1.	Constructing a prompt from goals, memory, and actions.
	2.	Sending it to the LLM (generate_response).
	3.	Parsing the LLM’s response to choose an action.
	4.	Executing that action in the environment.
	5.	Updating memory.
	6.	Checking if the agent should terminate.

3. language.py
	•	Translates the agent’s goals, memory, and available actions into a prompt the LLM can understand.
	•	Handles parsing the LLM’s response into tool calls (parse_response).

4. response_generator.py
	•	Calls the LLM (Cohere via LangChain) with a prompt.
	•	Handles tool/function calls, formats the result, applies timeout handling.
	•	Returns either text or a JSON representing which tool to call and its arguments.

5. tool_registry.py
	•	Dynamically registers Python functions as tools for the agent.
	•	Provides metadata (parameters, JSON schema, description, terminal flag, tags).
	•	Decorator @register_tool makes a function available to the agent.

⸻

Full flow:
	1.	User input: "List all CSV files in input_csvs".
	2.	Agent: set_current_task → memory updated.
	3.	Prompt built: AgentLanguage.construct_prompt() gathers goals, memory, actions.
	4.	LLM call: generate_response() → returns JSON like:
     5.	Agent parses response: get_action() → gets the corresponding Action.
	6.	Action executed: Environment.execute_action() → runs the Python function safely.
	7.	Memory updated: stores both the agent’s reasoning and environment result.
	8.	Check termination: repeat until either the agent calls terminate or max iterations reached.
