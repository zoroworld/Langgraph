from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
# for persitence
from langgraph.checkpoint.memory import MemorySaver
import os

load_dotenv()

# define model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# graph represent

# for persitence (RAM)
check_pointer = MemorySaver()

graph = StateGraph(ChatState)
def chat_node(state: ChatState):

    # take user query from state
    messages = state['messages']

    # send to llm
    response = model.invoke(messages)

    # response store state
    return {'messages': [response]}

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)
chatbot = graph.compile(checkpointer = check_pointer)
