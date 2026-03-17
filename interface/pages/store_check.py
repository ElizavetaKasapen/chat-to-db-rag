import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

st.title("Data storage check")

#TODO add/take to/from config
persist_directory = "./created_storages/demo"
embedding_model = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

contexts_store = Chroma(
    collection_name="contexts",
    persist_directory=persist_directory,
    embedding_function=embeddings
)

def get_facts_in_context(context_id: str):
    store = Chroma(
        collection_name=context_id,
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    docs = store.get(include=["documents"])
    return list(zip(docs["ids"], docs["documents"]))

def print_context():
    docs = contexts_store.get(include=["documents"])
    contexts = list(zip(docs["ids"], docs["documents"]))
    
    for cid, text in contexts:
        st.markdown(f"### 📂 Context `{cid}`")
        st.write(text)
        facts = get_facts_in_context(cid)
        st.markdown(f"**📘 Facts inside '{cid}':**")
        for fid, fact_text in facts:
            st.write(f"- {fid}: {fact_text}")

print_context()
