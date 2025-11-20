import os 
from dotenv import load_dotenv
from response_generator import generate_response
from GAME import Goal, Memory, Environment
from language import AgentFunctionCallingActionLanguage, PythonActionRegistry
from agent import Agent
import tools.file_tools
import tools.system_tools


load_dotenv()  # Load variables from .env file
api_key = os.getenv("COHERE_API_KEY")
os.environ["COHERE_API_KEY"] = api_key


# Define the agent's goals
goals = [
    Goal(priority=1,
         name="Explore Files",
         description="Navigate folders and list available CSV files."),
    Goal(priority=2,
         name="Analyze CSV Files",
         description="Count, identify, match, and inspect columns in CSV files using fuzzy matching."),
    Goal(priority=3,
         name="Clean and Consolidate Data",
         description="Clean CSV files and merge them into a consolidated report."),
    Goal(priority=4,
         name="Terminate",
         description="Call terminate when the user explicitly asks or the task is compelete."),
]


# Create the agent
agent = Agent(
    goals=goals,
    agent_language=AgentFunctionCallingActionLanguage(),
    action_registry=PythonActionRegistry(tags=["file_operations", "system"]),
    generate_response=generate_response,
    environment=Environment()
)


# Start interactive loop
memory = Memory()

print("")
print("")
print("======================================================================================================")
print("")
print("Hi, I'm your agent. Ask me to explore directories, list CSVs, or analyze files. Type 'exit' to quit.\n")

while True:
    user_input = input("🧑 You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    agent.set_current_task(memory, user_input)
    prompt = agent.construct_prompt(agent.goals, memory, agent.actions)
    response = agent.prompt_llm_for_action(prompt)

    action, invocation = agent.get_action(response)
    result = agent.environment.execute_action(action, invocation["args"])

    agent.update_memory(memory, response, result)
    
    if result.get("tool_executed"):
        print(f"\n🤖 Agent: {result['result']}\n")
    else:
        print(f"\n🤖 Agent encountered an error:\n{result['error']}\n")

    if agent.should_terminate(response):
        print("Agent has decided to terminate.")
        break