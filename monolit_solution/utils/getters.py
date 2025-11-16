from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .loaders import (
    load_qdrant_config,
    load_models_config,
    load_search_config,
    load_prompts,
)
from dotenv import load_dotenv

load_dotenv("chatbot.env")


# configuration getters
def get_qdrant_url():
    url = load_qdrant_config()["url"]
    return url

def get_qdrant_collection_name():
    return load_qdrant_config()["collection_name"]

def get_qdrant_vector_size():
     return load_qdrant_config()["vector_size"]

def get_doc_num_param():
    return load_search_config()["doc_num"]

def get_vectorstore_threshold_param():
    return load_search_config()["vectorstore_threshold"]

def get_llm_threshold_param():
    return load_search_config()["llm_threshold"]

#TODO add separate Embeddings model?
def get_models():
    model_config = load_models_config()
    model_origin = model_config["provider"]
    model_name = model_config[model_origin]["name"]
    if model_origin == "openai":
        embedding_function = OpenAIEmbeddings()
        llm = ChatOpenAI(model=model_name, temperature=0)
    elif model_origin == "ollama":
        llm = OllamaLLM(model=model_name, temperature=0)
        embedding_function = OllamaEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unsupported model origin: {model_origin}")
    return llm, embedding_function


# prompts getters
def get_classify_input_prompt():
    return load_prompts()["classify_input"]


def get_validate_statement_prompt():
    return load_prompts()["validate_statement"]


def get_check_duplicate_prompt():
    return load_prompts()["check_duplicate"]


def get_reformulate_for_db_prompt():
    return load_prompts()["reformulate_for_db"]


def get_handle_question_prompt():
    return load_prompts()["handle_question"]
