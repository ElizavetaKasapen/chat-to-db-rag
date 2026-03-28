from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import torch


load_dotenv("config/chatbot.env")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_llm(provider, model, temperature=0, context_size=None):

    extra_kwargs = {}

    # Ollama's context window size defaults to 4,096 tokens (or sometimes 2,048 depending on the model/version)
    if provider == "ollama" and context_size:
        extra_kwargs["num_ctx"] = context_size

    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        **extra_kwargs
    )


def init_embeddings(provider: str, model_name: str):
    try:
        if provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            device = get_device()

            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device}
            )

        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model_name)

        elif provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=model_name)

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    except ImportError:
        raise ImportError(
            f"Missing dependency for '{provider}'. "
            f"Install with: pip install langchain-{provider}"
        )