from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import numpy as np
from typing import Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings



def calculate_storage_redundancy(storage_path: str, embedding_function=None) -> Dict[str, Any]:
    """Calculate redundancy across all facts in a Chroma vectorstore.
    
    Args:
        storage_path: Path to the Chroma persist directory
        embedding_function: The embedding function used by the vectorstore (optional if already persisted)
        
    Returns:
        Dictionary with redundancy metrics
    """
    # Load the Chroma vectorstore
    vectorstore = Chroma(
        persist_directory=storage_path,
        embedding_function=embedding_function
    )
    
    # Get all documents and their embeddings
    collection = vectorstore._collection
    results = collection.get(include=["embeddings"])
    
    facts_embeddings = results["embeddings"]
    num_facts = len(facts_embeddings)
    
    if num_facts < 2:
        return {
            "num_facts": num_facts,
            "avg_similarity": 0.0,
            "max_similarity": 0.0,
            "min_similarity": 0.0,
            "std_similarity": 0.0,
            "high_similarity_pairs": 0,
            "medium_similarity_pairs": 0,
            "redundancy_ratio": 0.0,
            "pairwise_similarities": [],
            "total_pairs_compared": 0
        }
    
    similarities = []
    total_pairs = num_facts * (num_facts - 1) // 2
    
    for i, j in combinations(range(num_facts), 2):
        sim = cosine_similarity(
            [facts_embeddings[i]],
            [facts_embeddings[j]]
        )[0][0]
        similarities.append(float(sim))
    
    high_similarity_pairs = sum(1 for s in similarities if s > 0.87)
    medium_similarity_pairs = sum(1 for s in similarities if 0.7 < s <= 0.87)
    redundancy_ratio = high_similarity_pairs / total_pairs if total_pairs > 0 else 0
    
    return {
        "num_facts": num_facts,
        "avg_similarity": float(np.mean(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_similarity": float(np.min(similarities)),
        "std_similarity": float(np.std(similarities)),
        "high_similarity_pairs": high_similarity_pairs,
        "medium_similarity_pairs": medium_similarity_pairs,
        "redundancy_ratio": redundancy_ratio,
        "pairwise_similarities": similarities,
        "total_pairs_compared": total_pairs
    }

# embeddings = HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2",
#         model_kwargs={"device": "cuda"}
#     )
# metrics = calculate_storage_redundancy("./path/to/chroma_db", embedding_function=embeddings)

# Or if the vectorstore is already persisted with embeddings
metrics = calculate_storage_redundancy("benchmark_dataset_small_split")
import json
with open("evaluation_results\\reference_red_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)