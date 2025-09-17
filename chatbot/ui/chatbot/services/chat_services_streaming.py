import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# for persitence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

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
chatbot_graph = graph.compile(checkpointer=check_pointer)


# stream = chatbot_graph.stream(
#     {'messages': [HumanMessage(content="what is the recipe to make pasta")]},
#     config={"configurable": {"thread_id": "1"}},
#     stream_mode="messages"
# )

# def get_user_messages_servicese(user_message: str):
#     for message_chunk, metadata in chatbot_graph.stream(
#             {'messages': [HumanMessage(content=user_message)]},
#             config={"configurable": {"thread_id": "1"}},
#             stream_mode="messages",
#     ):
#         if message_chunk.content:
#             print(message_chunk.content, end=" ", flush=True)

# print(stream)

# def get_user_messages_servicese(user_message: str):
#     initial_state = {'messages': [HumanMessage(content=user_message)]}
#     config = {'configurable': {'thread_id': THREAD_ID}}
#     final_state = chatbot_graph.invoke(initial_state, config=config)
#     return final_state['messages'][-1].content
