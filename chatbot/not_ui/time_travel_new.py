from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
import os

# --- Load environment ---
load_dotenv()

# --- Define model ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# --- State definition ---
class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

# --- Nodes ---
def generate_joke(state: JokeState):
    prompt = f'generate a joke on the topic {state["topic"]}'
    response = model.invoke(prompt).content
    return {"joke": response}

def generate_explanation(state: JokeState):
    prompt = f'write an explanation for the joke - {state["joke"]}'
    response = model.invoke(prompt).content
    return {"explanation": response}

# --- Graph ---
graph = StateGraph(JokeState)
graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)

# --- Checkpointer ---
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# --- Run thread 1 ---
config1 = {"configurable": {"thread_id": "1"}}
final_state_1 = workflow.invoke({"topic": "pizza"}, config=config1)
print("Final state (thread 1):", final_state_1)

# --- Run thread 2 ---
config2 = {"configurable": {"thread_id": "2"}}
final_state_2 = workflow.invoke({"topic": "pasta"}, config=config2)
print("Final state (thread 2):", final_state_2)

# --- Inspect checkpoints for thread 1 ---
print("\n--- Checkpoint history for thread 1 ---")
history = list(workflow.get_state_history(config1))
for i, step in enumerate(history):
    print(f"Step {i}: checkpoint_id={step.checkpoint_id}, values={step.values}")

# --- Time travel: pick a checkpoint_id from history ---
checkpoint_id = history[1].checkpoint_id   # Example: after generate_joke
config_tt = {"configurable": {"thread_id": "1", "checkpoint_id": checkpoint_id}}

print("\n--- Time traveling to checkpoint ---")
resumed_state = workflow.invoke({}, config=config_tt)
print("Resumed state (from checkpoint):", resumed_state)
