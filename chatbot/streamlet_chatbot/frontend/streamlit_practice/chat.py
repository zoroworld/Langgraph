import streamlit as st

def intro():
    with st.chat_message('user'):
        st.text('Hi')
    with st.chat_message('assistant'):
        st.text('How can i help you')

    user_input = st.chat_input("Type Here...")

    if user_input:
        with st.chat_message('user'):
            st.text(user_input)

intro()