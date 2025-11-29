
from langchain_community.vectorstores import Chroma
import numpy as np


def search_similar(vectordb, query, min_similarity=0.85, k=20, topic=None):
    """
    Uses Chroma's native cosine similarity.
    relevance_score is already cosine similarity in [0, 1].
    """
    if topic:
        results = vectordb.similarity_search_with_relevance_scores( #returns SIMILARITY, the check is correct
            query, k=k, filter={"folder_name": topic})
    else:
        results = vectordb.similarity_search_with_relevance_scores(query, k=k)

    filtered = []
    for doc, score in results:
        if score >= min_similarity:
            filtered.append((doc, score))

    return filtered

def save_matches(matches):
    output_file = "matches_output.txt"

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"Found {len(matches)} matches\n\n")
        
        for i, (doc, sim) in enumerate(matches):
            f.write(f"--- Match {i+1} (similarity={sim:.4f}) ---\n")
            f.write(f"Metadata: {doc.metadata}\n")
            f.write(f"Text: {doc.page_content}\n\n")




def calculate_frequency(topic, query, embeddings=None, min_similarity = 0.70, benchmark_persist_directory = "benchmark_dataset_small_split"):    
    vectordb = Chroma(
        embedding_function=embeddings, #TODO do I really need it here?
        persist_directory=benchmark_persist_directory
    )
    print(
                f"\n🔍 Searching for: \"{query}\" (similarity ≥ {min_similarity})")
    matches = search_similar(
                vectordb, query, min_similarity=min_similarity, topic=topic) #here I already search by topic
    if matches: 
        save_matches(matches)
        # Simple linear scaling
        freq_match_scaled = np.clip(len(matches) / 5, 0, 1) # Interpretation: 5 matches/topic = perfect score #TODO change from 5 to avg number of articles in topics

        return freq_match_scaled
    else: return 0.0
    
