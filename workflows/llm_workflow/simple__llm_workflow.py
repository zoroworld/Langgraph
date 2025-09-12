from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict
from IPython.display import Image
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Step 1 :  setup of hugging face=====================================
# Load environment variables from .env
load_dotenv()
api_key = os.getenv("HF_TOKEN")
# Optional: If you want to use the raw HF InferenceClient
client = InferenceClient(
    provider="auto",
    token=api_key,
)

# Step 2: create models=================================================
model_id = "moonshotai/Kimi-K2-Instruct"

# HuggingFaceEndpoint for LangChain
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
)

# Wrap as Chat Model
model = ChatHuggingFace(llm=llm)

# Step 3: create a state

class LLMState(TypedDict):
    question: str
    answer: str





# Step 4: create our graph
graph = StateGraph(LLMState)

# create nodes
def llm_qa(state: LLMState) -> LLMState:
    # extract the question from state
    question = state['question']
    # form a prompt
    prompt = f'Answer the following question {question}'
    # ask that question to the LLM
    answer = model.invoke(prompt).content
    # update the answer in the state
    state['answer'] = answer
    return state

graph.add_node('llm_qa', llm_qa) # (node, function)


# create edges
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)



# Step 5: compile
workflow = graph.compile()



# Step 6: execute
intial_state = {'question': 'How far is moon from the earth?'}
final_state = workflow.invoke(intial_state)

# Step 7: ouput
print(final_state['answer'])


# Step 8: model result for check (wihot graph)
result = model.invoke('How far is moon from the earth?').content
print(result)