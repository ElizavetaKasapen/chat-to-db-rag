"""
interface.py

Streamlit UI for the Knowledge Base Chat.
"""

import streamlit as st
from core.workflow import process_user_input



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

# Get user input
user_input = st.chat_input("Enter your story or question...") #TODO limit input, add possibility to add docs

if user_input:
    # Append user's message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call workflow to process input and get AI response
    answer = process_user_input(user_input, st.session_state.messages)

    # Append AI response to session state
    st.session_state.messages.append({"role": "ai", "content": answer})

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
