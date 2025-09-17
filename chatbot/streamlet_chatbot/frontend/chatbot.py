import streamlit as st
from langchain_core.messages import HumanMessage
import sys, os
# Add project root (resume_chatbot/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.backend_chatbot import chatbot


config = {'configurable': {'thread_id': "thread-1"}}


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []




def intro():
    user_input = st.chat_input("Type Here...")

    if user_input:
        st.session_state['message_history'].append({'role':'user', 'content':user_input})
        with st.chat_message('user'):
            st.text(user_input)


        # for ai work


        final_state = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)
        bot_response = final_state["messages"][-1].content

        st.session_state['message_history'].append({'role': 'assistant', 'content':bot_response })
        with st.chat_message('assistant'):
            st.text(bot_response)


# message history loading speaking..
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# current text message
intro()