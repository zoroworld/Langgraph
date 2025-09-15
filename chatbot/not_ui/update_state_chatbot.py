from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
import os

load_dotenv()

# define model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

#state
class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state: JokeState):

    prompt = f'generate a joke on the topic {state["topic"]}'
    response = model.invoke(prompt).content

    return {'joke': response}

def generate_explanation(state: JokeState):

    prompt = f'write an explanation for the joke - {state["joke"]}'
    response = model.invoke(prompt).content

    return {'explanation': response}

graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config1 = {"configurable": {"thread_id": "1"}}
final_state_1 = workflow.invoke({'topic':'pizza'}, config=config1)
# print(final_state_1)
# print(workflow.get_state(config1))
# print(list(workflow.get_state_history(config1)))

config2 = {"configurable": {"thread_id": "2"}}
final_state_2 =  workflow.invoke({'topic':'pasta'}, config=config2)
#print(final_state_2)
# print(workflow.get_state(config1))
# print(list(workflow.get_state_history(config2)))



# time travel not give thread point also checkpoint (intrupt)
# print(workflow.get_state({"configurable": {"thread_id": "1", "checkpoint_id": "1f06cc6e-7232-6cb1-8000-f71609e6cec5"}}))

# time travel new

print(workflow.invoke(None, {"configurable": {"thread_id": "1", "checkpoint_id": "1f06cc6e-7232-6cb1-8000-f71609e6cec5"}}))
# list(workflow.get_state_history(config1))

#update state

workflow.update_state({"configurable": {"thread_id": "1", "checkpoint_id": "1f06cc6e-7232-6cb1-8000-f71609e6cec5", "checkpoint_ns": ""}}, {'topic':'samosa'})
state_history_list_1 = list(workflow.get_state_history(config1))
print(state_history_list_1)
workflow.invoke(None, {"configurable": {"thread_id": "1", "checkpoint_id": "1f06cc72-ca16-6359-8001-7eea05e07dd2"}})
state_history_list_2 =list(workflow.get_state_history(config1))
print(state_history_list_2)
