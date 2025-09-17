import streamlit as st
from langchain_core.messages import HumanMessage
import sys, os
# Add project root (resume_chatbot/) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.backend_chatbot import chatbot



if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []




def intro():
    user_input = st.chat_input("Type Here...")

    if user_input:
        st.session_state['message_history'].append({'role':'user', 'content':user_input})
        with st.chat_message('user'):
            st.text(user_input)


        # for ai work
        with st.chat_message('assistant'):
            ai_message = st.write_stream(
                message_chunk.content for message_chunk , metadata in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config={'configurable': {'thread_id': "thread-1"}},
                    stream_mode='messages'
                )
            )
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message })



# message history loading speaking..
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# current text message
intro()