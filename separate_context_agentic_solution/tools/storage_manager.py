from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from .context_representation import ContextRepresentation
import uuid
import numpy as np

class ContextFactManager:
    def __init__(self, persist_directory="./chroma_db_separate_context", 
                 embedding_model="all-MiniLM-L6-v2",
                 context_representation = "concat"):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self._contexts_store = self._get_contexts_store()
        self._context_representation = ContextRepresentation(context_representation)

    def _get_contexts_store(self):
        embedding_fn = HuggingFaceEmbeddings(model_name=self.embedding_model)
        return Chroma(
            collection_name="contexts",
            persist_directory=self.persist_directory,
            embedding_function=embedding_fn
        )

    def _get_facts_store(self, context_id):
        embedding_fn = HuggingFaceEmbeddings(model_name=self.embedding_model)
        return Chroma(
            collection_name=context_id,
            persist_directory=self.persist_directory,
            embedding_function=embedding_fn
        )

    def _update_context(self, context_id: str):
        facts_store = self._get_facts_store(context_id)
        updated_text = self._context_representation.build(facts_store)
         # Prepare the Document depending on type
        if isinstance(updated_text, np.ndarray):
            # Mean embedding: store in metadata, keep page_content empty
            doc = Document(
                page_content="", 
                metadata={
                    "context_id": context_id,
                    "embedding": updated_text.tolist()
                }
            )
        elif isinstance(updated_text, str):
            # Textual representation: store in page_content
            doc = Document(
                page_content=updated_text,
                metadata={"context_id": context_id}
            )
        else:
            raise TypeError(
                f"Unexpected context representation type: {type(updated_text)}"
            )
        self._contexts_store.update_document(
            document_id=context_id,
            document=doc
        )
        # self._contexts_store.update_document(
        #     document_id=context_id,
        #     document=Document(updated_text, metadata={
        #                         "context_id": context_id})
        # )

    def delete_context(self, context_id: str) -> str:
        """Delete an entire context and all its related facts."""
        # Delete the context document from the main context store
        self._contexts_store.delete(ids=[context_id])

        # Delete the associated facts collection
        facts_store = self._get_facts_store(context_id)
        facts_store.delete_collection()

        return f"Context '{context_id}' and all related facts were permanently deleted."

    def create_context(self, fact_text: str) -> str:
        """Create a new context with an initial fact. Returns context_id."""
        context_id = str(uuid.uuid4())
        facts_store = self._get_facts_store(context_id)
        facts_store.add_texts(texts=[fact_text])
        self._contexts_store.add_texts(
            texts=[fact_text],
            metadatas=[{"context_id": context_id}],
            ids=[context_id]
        )
        return context_id

    def add_fact(self, context_id: str, fact_text: str) -> str:
        """Add a new fact to a context, and update the context document."""
        facts_store = self._get_facts_store(context_id)
        facts_store.add_texts(texts=[fact_text])
        self._update_context(context_id)
        return f"Added fact '{fact_text}' to context {context_id}"

    def update_fact(self, context_id, fact_id, updated_fact):
        facts_store = self._get_facts_store(context_id)
        facts_store.update_document(
            document_id=fact_id,
            document=Document(
                updated_fact,
                metadata={"updated": True}
            )
        )
        self._update_context(context_id)
        return f"Fact '{fact_id}' and its context were updated."

    def delete_fact(self, context_id: str, fact_id: str) -> str:
        """Delete a fact from a context; if no more facts, delete the context too."""
        facts_store = self._get_facts_store(context_id)
        facts_store.delete(ids=[fact_id])
        remaining = facts_store.get()
        if not remaining["ids"]:
            self._contexts_store.delete(ids=[context_id])
            facts_store.delete_collection()
            return f"Context '{context_id}' and all related facts were deleted."
        else:
            self._update_context(context_id)
            return f"Fact '{fact_id}' deleted and context '{context_id}' updated."

    def context_similarity_search(self, query: str, k: int = 1) -> str:
        """Search among contexts and return context_id most similar to the query."""
        results = self._contexts_store.similarity_search(query, k=k)
        if results:
            return results[0].metadata["context_id"]
        else:
            return "no_context_found"

    def fact_similarity_search(self, context_id: str, fact_text: str, top_k: int = 5):
        """Search for similar facts within a given context."""
        facts_store = self._get_facts_store(context_id)
        return facts_store.similarity_search(fact_text, k=top_k)

    def get_contexts(self):
        """Return list of (context_id, text) of all contexts."""
        docs = self._contexts_store.get(include=["documents"])
        return [(cid, doc) for cid, doc in zip(docs["ids"], docs["documents"])]

    def get_facts_in_context(self, context_id: str):
        """Return list of (fact_id, text) for facts in a context."""
        facts_store = self._get_facts_store(context_id)
        docs = facts_store.get(include=["documents"])
        return [(fid, doc) for fid, doc in zip(docs["ids"], docs["documents"])]
