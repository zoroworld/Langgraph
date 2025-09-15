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


# initial_state = {
#     'messages': [HumanMessage(content='What is the capital of india')]
# }
#
# final_state = chatbot.invoke(initial_state)

# print(final_state)

# {
#   'messages': [
#   HumanMessage(content='What is the capital of india',
#   additional_kwargs={},
#   response_metadata={},
#   id='dd5bd9f5-ab05-4093-a7ce-9ca2392a4e79'),
#   AIMessage(content='The capital of India is **New Delhi**.',
#   additional_kwargs={},
#   response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': []}, id='run--d3b7adfd-1be6-45f5-b460-633b4fabd248-0',
#   usage_metadata={'input_tokens': 7, 'output_tokens': 26, 'total_tokens': 33, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 17}})
#   ]
#  }

# extract
# print(final_state["messages"][-1].content)

#  using checkpointer use thread
thread_id = "1"

while True:
    user_message = input("Type here: ")
    print("User: ", user_message)
    user_message = user_message.strip().lower()
    if user_message in ['exit', 'quit', 'bye']:
        break

    initial_state = {
        'messages': [HumanMessage(content=user_message)]
    }

    config ={'configurable': {'thread_id':thread_id}}

    final_state = chatbot.invoke(initial_state, config=config)
    response = final_state["messages"][-1].content


    print('Ai: ', response)

chat_result = chatbot.get_state(config=config)
print(chat_result)

# StateSnapshot(values={'messages': [
#     HumanMessage(content='mi my name is manish',
#                  additional_kwargs={},
#                  response_metadata={},
#                  id='79bde709-815a-42bb-a793-6bfd0346e96b'),
#     AIMessage(content='Hi Manish, nice to meet you!',
#               additional_kwargs={},
#               response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': []}, id='run--09bad2c5-9417-4713-87dc-43a237b5b3be-0', usage_metadata={'input_tokens': 7, 'output_tokens': 152, 'total_tokens': 159, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 144}}),
#     HumanMessage(content='what is my name',
#                  additional_kwargs={},
#                  response_metadata={},
#                  id='ad0dffd5-3062-4eee-86ad-811c3ad0d114'),
#     AIMessage(content='Your name is Manish.',
#               additional_kwargs={},
#               response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': []},
#               id='run--114b2ea8-5e18-48d0-8281-2ce1f27188b6-0', usage_metadata={'input_tokens': 21, 'output_tokens': 25, 'total_tokens': 46, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 20}}),
#     HumanMessage(content='sum from 1 to 100',
#                  additional_kwargs={},
#                  response_metadata={},
#                  id='d76d3604-65cf-4e3f-8310-e1e48a62b741'),
#     AIMessage(content='The sum of numbers from 1 to 100 is **5050**.\n\nThis can be calculated using the formula for the sum of an arithmetic series:\nSum = n * (first term + last term) / 2\nWhere n = 100, first term = 1, last term = 100.\n\nSum = 100 * (1 + 100) / 2\nSum = 100 * 101 / 2\nSum = 50 * 101\nSum = 5050',
#               additional_kwargs={},
#               response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': []},
#               id='run--1467263f-c8d1-431e-b35f-494f6604795d-0',
#               usage_metadata={'input_tokens': 37, 'output_tokens': 331, 'total_tokens': 368, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 205}}),
#     HumanMessage(content='multiply with 2',
#                 additional_kwargs={},
#                  response_metadata={},
#                  id='21d73717-4f78-4a28-904b-4c5da11e73ef'),
#     AIMessage(content="Okay, let's multiply 5050 by 2:\n\n5050 * 2 = **10100**",
#               additional_kwargs={},
#               response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': []},
#               id='run--bb94735c-29cf-4340-83ba-71259762ae29-0',
#               usage_metadata={'input_tokens': 169, 'output_tokens': 53, 'total_tokens': 222, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 22}})]},
#     next=(),
#     config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f091fee-c4ef-6aba-800a-0ec01f1d1f2b'}},
#     metadata={'source': 'loop', 'step': 10, 'parents': {}},
#     created_at='2025-09-15T06:40:52.080472+00:00',
#     parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f091fee-b1ab-6c99-8009-d02f52dbd558'}},
#     tasks=(),
#     interrupts=())
