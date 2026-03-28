"""
context_representation.py

Defines the ContextRepresentation class, which generates a single 
representation (embedding, summary, or extracted entities)
from a collection of facts.

Supports two input modes:
- Chroma store (VectorStore)
- List[str]
"""

from typing import Callable, List, Union, Dict
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from utils.utils import init_llm, init_embeddings
from config.config import get_storage_config #TODO check import 

from dotenv import load_dotenv

load_dotenv("config/chatbot.env")

storage_config = get_storage_config()


class ContextRepresentation:
    """
    Generate a uniform representation for a set of documents/facts.

    Supported strategies:
        - "mean": mean embedding representation
        - "main_objects": extract main entities from the text
        - "summary": LLM-generated concise summary
    """

    def __init__(self, context_type: str, embedding_config, llm_config=None): 
        """
        Initialize context representation builder.

        Args:
            context_type: Representation type ("mean", "summary", etc.)
            embedding_model: HuggingFace embedding model name
        """
        self.context_type = context_type
        if llm_config: self.llm = init_llm(**llm_config)

        # Embedding model for list[str] representation
        self.embedding_fn = init_embeddings(**embedding_config)

        # Strategy maps
        self._store_strategies: Dict[str, Callable] = {
            "mean": self._mean_store,
            "main_objects": self._main_objects_store,
            "summary": self._summary_store,
        }

        self._list_strategies: Dict[str, Callable] = {
            "mean": self._mean_list,
            "main_objects": self._main_objects_list,
            "summary": self._summary_list,
        }


    # Public API

    def build(self, context: Union[list, object]):
        """
        Build the representation from either list[str] or vector store.

        Args:
            context: list of text documents, or Chroma vector store

        Returns:
            Representation (str or np.ndarray)
        """
        if isinstance(context, list):
            return self.build_from_list(context)

        return self.build_from_store(context)

    def build_from_store(self, store):
        """Build representation using a vector store."""
        strategy = self._store_strategies.get(self.context_type, self._default_store)
        return strategy(store)

    def build_from_list(self, docs: List[str]):
        """Build representation from list of raw strings."""
        strategy = self._list_strategies.get(self.context_type)
        if strategy is None:
            return " ".join(docs)
        return strategy(docs)


    # Mean Embedding Representations

    def _mean_store(self, store):
        """Compute mean embedding from vector store embeddings."""
        data = store.get(include=["embeddings"])
        vectors = data.get("embeddings", [])
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        return None

    def _mean_list(self, docs: List[str]):
        """Compute mean embedding from list[str]."""
        if not docs:
            return None
        vectors = self.embedding_fn.embed_documents(docs)
        return np.mean(vectors, axis=0) if vectors else None


    # Object Extraction

    def _main_objects_store(self, store):
        data = store.get(include=["documents"])
        return self._extract_main_objects(" ".join(data["documents"]))

    def _main_objects_list(self, docs):
        return self._extract_main_objects(" ".join(docs))

    def _extract_main_objects(self, text: str) -> str:
        """LLM task: extract key entities."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Extract key entities (people, places, organizations, objects, times) "
                "from the text:\n\n{text}\n\nReturn a concise comma-separated list."
            )
        )
        response = self.llm.invoke(prompt.format(text=text))
        return response.content


    # Summary

    def _summary_store(self, store):
        data = store.get(include=["documents"])
        return self._summarize(" ".join(data["documents"]))

    def _summary_list(self, docs):
        return self._summarize(" ".join(docs))

    def _summarize(self, text: str) -> str:
        """LLM task: generate concise summary."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text concisely:\n\n{text}"
        )
        response = self.llm.invoke(prompt.format(text=text))
        return response.content


    # Default fallback

    def _default_store(self, store):
        """Fallback: return concatenated documents."""
        data = store.get(include=["documents"])
        return " ".join(data["documents"])
