"""
interface.py

Streamlit UI for the Knowledge Base Chat.
"""

import streamlit as st
from core.workflow import process_user_input

#TODO delete later, it's just to test 
from evaluate.get_mapping import extract_facts_dict

st.title("Knowledge Base Chat")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add initial greeting if empty
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "ai",
        "content": "Hello! How can I help you today?"
    })

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Enter your story or question...") #TODO limit input, add possibility to add docs

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)
    with st.spinner("Thinking..."):
        answer = process_user_input(user_input, st.session_state.messages)
    #extarcted_facts = extract_facts_dict(answer, "test")
    st.session_state.messages.append({"role": "ai", "content": answer})
   # st.session_state.messages.append({"role": "ai", "content": extarcted_facts})
    st.rerun()

   # st.chat_message(msg["role"]).write(msg["content"])
