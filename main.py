import streamlit as st
from core import *
# Interface

st.title("Knowledge Base Chat")
# RN no real memory, just to see previous input for me TODO add mem
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        "role": "ai",
        "content": "Hello! How can I help you today?"
    })


# TODO del logs for final version
user_input = st.chat_input("Enter a statement or question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    logging.info(f"User's input: {user_input}")
    input_type = classify_input(user_input)
    logging.info(f"Input classified as : {input_type}")
    if input_type == "statement":
        if validate_statement(user_input):
            if not check_duplicate(user_input, threshold=LLM_THRESHOLD):
                structured_data = reformulate_for_db(user_input)
                store_manager.vectorstore.add_texts([structured_data])
                logging.info("Total docs:", store_manager.count_documents())
                st.session_state.messages.append({"role": "ai", "content": f"Statement added to DB! Here is reformulated statement: {structured_data}"})
                
            else:
                logging.info("This information already exists in the database.")
                st.session_state.messages.append({"role": "ai", "content": "This information already exists in the database."})
        else:
            logging.info("This statement seems invalid or implausible.")
            st.session_state.messages.append({"role": "ai", "content": "This statement seems invalid or implausible."})
    elif input_type == "question":
        answer = handle_question(user_input)
        logging.info(f"Question handled and answered. Answer: {answer}")
        st.session_state.messages.append({"role": "ai", "content": answer})

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
