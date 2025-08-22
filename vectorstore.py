from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore


class QdrantStoreManager:
    
    def __init__(self, url: str, collection_name: str, embedding_function, vector_size: int = 1536):#768):
        """
        Initialize and prepare a Qdrant vectorstore.
        Creates the collection if it does not exist.
        """
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        # Ensure collection exists
        collections = self.client.get_collections().collections
        existing_names = [col.name for col in collections]

        if self.collection_name not in existing_names:
            print(f"Collection '{self.collection_name}' does not exist. Creating it...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        else:
            print(f"Collection '{self.collection_name}' already exists. Skipping creation.")

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_function
        )

        print(f"Number of documents in '{self.collection_name}': {self.count_documents()}")


    def count_documents(self) -> int:
        """Return the number of documents in the collection."""
        return self.client.count(
            collection_name=self.collection_name,
            exact=True
        ).count

    
    def similarity_search(self, query: str, k: int = 5, vectorstore_threshold: float = 0.7):
        """Search for the most similar documents to a query."""
        return self.vectorstore.similarity_search(query, k=k, score_threshold = vectorstore_threshold)

