import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from vectorstore import QdrantStoreManager
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv("chat-to-db.env")

#TODO make them as "globals"
# llm = HuggingFaceEndpoint(
#     model="meta-llama/Llama-3.1-8B",
#     task="conversational",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     provider="auto",  # let Hugging Face choose the best provider for you
# )

# chat_model = ChatHuggingFace(llm=llm)
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model= "gpt-3.5-turbo", temperature=0)   

# model = "llama3.1:8b"
# llm = ChatOllama(base_url="http://localhost:11434", model=model)

# Initialize once
# TODO get from config
#@st.cache_resource TODO call it once
store_manager = QdrantStoreManager(
    url="http://localhost:6333",
    collection_name="test_chat-to-db_gpt",
    embedding_function=embeddings
)

print("Total docs:", store_manager.count_documents())

def classify_input(user_input):
    """Classify input as question or statement"""
    prompt = f"Is the following input a question or a statement? Answer only 'question' or 'statement':\n\n{user_input}"
    response = llm.invoke(prompt).content
    return response.lower().strip()

def validate_statement(statement):
    """Check if statement is potentially valid"""
    prompt = f"Check if this statement is plausible: '{statement}'. Answer 'valid' or 'invalid'."
    response = llm.invoke(prompt).content
    return response.lower().strip() == "valid" #TODO maybe redo this

def check_duplicate(statement, threshold=0.8):
    """Check if statement is already in DB"""
    results = store_manager.similarity_search(statement, k=5)
    for r in results:
        # Use LLM to compare similarity semantically
        prompt = f"Compare the following statements and give a similarity score 0-1:\n1) {statement}\n2) {r.page_content}. Return ONLY the score."
        score = float(llm.invoke(prompt).content)  # Assume LLM outputs a numeric score
        if score >= threshold:
            return True
    return False


def reformulate_for_db(statement):
    """Reformulate statement into canonical DB format"""
    prompt = f"Rephrase this statement in a clear, standardized way for database storage: '{statement}'"
    response = llm.invoke(prompt).content
    return response  

def handle_question(question):
    """Answer question using DB + LLM"""
    results = store_manager.similarity_search(question, k=5)
    context = "\n".join([r.page_content for r in results])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content

#Interface 

st.title("Knowledge Base Chat")
#RN no real memory, just to see previous input for me TODO add mem
# msgs = StreamlitChatMessageHistory(key="test") 
# if len(msgs.messages) == 0:
#     st.chat_message("ai").write("How can I help you?")
# else: 
#     for msg in msgs.messages:
#         st.chat_message(msg.type).write(msg.content)


user_input = st.chat_input("Enter a statement or question...")  
st.chat_message("user").write(user_input)
if user_input:
    input_type = classify_input(user_input)

    if input_type == "statement":
        if validate_statement(user_input):
            if not check_duplicate(user_input):
                structured_data = reformulate_for_db(user_input)
                store_manager.vectorstore.add_texts([structured_data])
                st.chat_message("ai").write("Statement added to DB!")
            else:
                st.chat_message("ai").write("This information already exists in the database.")
        else:
            st.chat_message("ai").write("This statement seems invalid or implausible.")
    
    elif input_type == "question":
        answer = handle_question(user_input)
        st.chat_message("ai").write(answer)
