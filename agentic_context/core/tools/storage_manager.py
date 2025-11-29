"""
storage_manager.py

Provides ContextFactManager, an abstraction over Chroma to store:
- contexts (summaries or embeddings)
- facts belonging to each context

Handles CRUD operations and automatic context regeneration.
"""

from typing import List, Tuple, Union
import uuid
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from .context_representation import ContextRepresentation


class ContextFactManager:
    """
    Manages contexts and their associated facts using Chroma DB.

    Responsibilities:
        - Create new contexts
        - Add, update, delete facts
        - Maintain derived context representation (summary/embedding)
        - Similarity search across contexts and facts
    """

    def __init__(
        self,
        persist_directory="./chroma_db_separate_context",
        embedding_model="all-MiniLM-L6-v2",
        context_representation: ContextRepresentation = None
    ):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self._contexts_store = self._init_contexts_store()
        self._context_representation = (
            context_representation or ContextRepresentation("summary")
        )


    # Store initialization

    def _init_contexts_store(self):
        """Chroma collection holding one entry per context."""
        return Chroma(
            collection_name="contexts",
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"} #TURNED ALL TO COSINE SIMILARITY
        )

    def _get_facts_store(self, context_id: str):
        """Return Chroma store containing all facts for given context."""
        return Chroma(
            collection_name=context_id,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"} #TURNED ALL TO COSINE SIMILARITY
        )


    # Context Lifecycle

    def create_context(self, fact_texts: List[str], representation): 
        """
        Create a new context with initial facts.

        Args:
            fact_texts: list of fact strings
            representation: np.ndarray (embedding) or str (summary)

        Returns:
            context_id
        """
        context_id = str(uuid.uuid4())
        facts_store = self._get_facts_store(context_id)
        facts_store.add_texts(texts=fact_texts)
        print(f"Added facts to fact store context_id: {context_id}, facts {fact_texts}")
        if isinstance(representation, np.ndarray):
            self._contexts_store._collection.add(
                ids=[context_id],
                documents=[f"Mean embedding of '{context_id}'"],
                embeddings=[representation]
            )
        elif isinstance(representation, str):
            self._contexts_store.add_texts(
                ids=[context_id],
                texts=[representation],
            )
        else:
            raise TypeError(
                f"Context representation must be str or np.ndarray, got {type(representation)}"
            )
        return context_id
    

    def delete_context(self, context_id: str) -> str:
        """Delete context document and its fact collection."""
        self._contexts_store.delete(ids=[context_id])
        self._get_facts_store(context_id).delete_collection()
        return f"Context '{context_id}' deleted."


    # Fact CRUD

    def add_fact(self, context_id: str, fact_text: str) -> str:
        """Add fact and regenerate context representation."""
        facts = self._get_facts_store(context_id)
        facts.add_texts([fact_text])
        self._update_context(context_id)
        return f"Fact added to context {context_id}"

    def update_fact(self, context_id: str, fact_id: str, new_text: str) -> str:
        """Modify a fact and recompute context summary."""
        facts = self._get_facts_store(context_id)
        facts.update_document(
            document_id=fact_id,
            document=Document(new_text, metadata={"updated": True})
        )
        self._update_context(context_id)
        return f"Fact '{fact_id}' updated."

    def delete_fact(self, context_id: str, fact_id: str) -> str:
        """Delete fact; delete context if empty."""
        facts = self._get_facts_store(context_id)
        facts.delete(ids=[fact_id])

        if not facts.get()["ids"]:
            return self.delete_context(context_id)

        self._update_context(context_id)
        return f"Fact '{fact_id}' deleted."


    # Context Regeneration

    def _update_context(self, context_id: str):
        """Recompute representation for context based on all facts."""
        facts_store = self._get_facts_store(context_id)
        representation = self._context_representation.build(facts_store)
        if isinstance(representation, np.ndarray):
            self._contexts_store._collection.update(
                ids=context_id,
                documents=f"Mean embedding for {context_id}",
                embeddings=representation
            )
        elif isinstance(representation, str):
            doc = Document(
                page_content=representation,
                metadata={"context_id": context_id}
            )
            self._contexts_store.update_document(
                document_id=context_id,
                document=doc
            )
        else:
            raise TypeError(f"Invalid representation type: {type(representation)}")


    # Search

    def context_similarity_search(self, query, k=1, threshold=0.4): #TODO move to config - do 0.4 for retrieving data, 0.2 for memory manager
        """Return context_id most similar to query (text or vector)."""
        if isinstance(query, np.ndarray):
           results = self._contexts_store.similarity_search_by_vector_with_relevance_scores( query.tolist(), k=k) #Returns Distance. Lower score represents more similarity.

        else:
           results = self._contexts_store.similarity_search_with_score(query, k=k) # Returns also distance. Lower score represents more similarity.
        if not results:
            return None
        print(f"RESULTS: { results}")
        doc, score = results[0]
        print(f"SIMILARITY SCORE: {score}")
        if score <= threshold:  
            print(f"RETURNED ID: {doc.id}")
            return doc.id 
        return None


    def fact_similarity_search(self, context_id: str, query: str, k=5):
        """Find most similar facts inside a given context."""
        facts = self._get_facts_store(context_id)
        return facts.similarity_search(query, k=k)


    # Queries

    def get_contexts(self) -> List[Tuple[str, str]]:
        """Return all context IDs with their stored text representation."""
        docs = self._contexts_store.get(include=["documents"])
        return list(zip(docs["ids"], docs["documents"])) 

    def get_facts_in_context(self, context_id: str):
        """Return all facts in a given context."""
        store = self._get_facts_store(context_id)
        docs = store.get(include=["documents"])
        return list(zip(docs["ids"], docs["documents"]))
    

    def get_facts_embeddings_in_context(self, context_id: str):
        """Return all facts in a given context."""
        store = self._get_facts_store(context_id)
        docs = store.get(include=["embeddings"])
        return np.asarray(docs["embeddings"])


    def get_storage(self):
        contexts = self.get_contexts()
        for cid, text in contexts:
            print(f" \n  ***Context {cid} : {text}***")
            facts = self.get_facts_in_context(cid)
            print(f"    \n *Facts inside '{cid}' context:*")
            for fid, text in facts:
                print(f"  - {fid} : {text}")

    def get_context_embedding(self, context_id: str) -> np.ndarray | None:
        """Return the embedding for a given context."""
        result = self._contexts_store.get(
            ids=[context_id],
            include=["embeddings"]
        )

        if not result["ids"] or result["embeddings"][0] is None:
            return None

        return np.asarray(result["embeddings"][0])
