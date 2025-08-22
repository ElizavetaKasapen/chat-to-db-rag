import streamlit as st
from vectorstore import QdrantStoreManager
import logging
from utils.getters import *

logging.basicConfig(
    level=logging.INFO, format=f"%(asctime)s %(levelname)s %(name)s : %(message)s"
)

VECTORSTORE_THRESHOLD = get_vectorstore_threshold_param() 
LLM_THRESHOLD = get_llm_threshold_param()
DOC_NUM = get_doc_num_param()  # for similarity search

llm, embeddings = get_models()


# Initialize once
@st.cache_resource
def get_store_manager():
    store_url = get_qdrant_url()
    store_collection = get_qdrant_collection_name()
    vector_size = get_qdrant_vector_size()
    store_manager = QdrantStoreManager(
        url=store_url,
        collection_name=store_collection,
        embedding_function=embeddings,
        vector_size=vector_size,
    )
    print("Total docs:", store_manager.count_documents())
    return store_manager


store_manager = get_store_manager()

# helper functions
def get_response(prompt):
    response = llm.invoke(prompt)#.content
    response = getattr(response, "content", response).lower().strip()
    return response


def run_prompt(prompt_template, **kwargs):
    """
    Unified function to handle prompts for any task.
    """
    # safely escape curly braces in user inputs
    safe_kwargs = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in kwargs.items()}
    prompt = prompt_template.format(**safe_kwargs)
    return get_response(prompt)


def classify_input(user_input):
    """Classify input as question or statement"""
    prompt_template = get_classify_input_prompt()
    return run_prompt(prompt_template, user_input=user_input)


def validate_statement(statement):
    """Check if statement is potentially valid"""
    prompt_template = get_validate_statement_prompt()
    response = run_prompt(prompt_template, statement=statement)
    logging.info(response)
    if response not in ["valid", "invalid"]:
        logging.info(f"Unexpected LLM response: '{response}'. Defaulting to 'invalid'.")
        return False
    return response == "valid"  # TODO maybe redo this


def check_duplicate(statement, threshold=0.8):
    """Check if statement is already in DB"""
    results = store_manager.similarity_search(statement, k=DOC_NUM, vectorstore_threshold=VECTORSTORE_THRESHOLD) #gets the most simialr docs from DB
    for r in results:
        # Use LLM to compare similarity semantically, so we will not have the same info in DB twise
        prompt_template = get_check_duplicate_prompt()
        response = run_prompt(prompt_template, statement=statement, page_content=r.page_content)
        score = float(response)  # Assume LLM outputs a numeric score
        logging.info(f"score: {score}")
        if score >= threshold:
            return True
    return False


def reformulate_for_db(statement):
    """Reformulate statement into canonical DB format"""
    prompt_template = get_reformulate_for_db_prompt()
    return run_prompt(prompt_template, statement=statement)


def handle_question(question):
    """Answer question using DB + LLM"""
    results = store_manager.similarity_search(question, k=DOC_NUM, vectorstore_threshold=VECTORSTORE_THRESHOLD)
    context = "\n".join([r.page_content for r in results])
    prompt_template = get_handle_question_prompt()
    logging.info(f"context: {context}")
    return run_prompt(prompt_template, context=context, question=question)