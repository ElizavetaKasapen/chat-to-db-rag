import os
from langchain_community.vectorstores import Chroma
import numpy as np


def count_topic_articles(topic, dataset_path="dataset"):
    topic_path = os.path.join(dataset_path, topic)
    
    if not os.path.exists(topic_path):
        return 0
    
    return len([
        f for f in os.listdir(topic_path)
        if os.path.isfile(os.path.join(topic_path, f))
    ])



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




def calculate_frequency(topic, query, embeddings=None, min_similarity = 0.80, benchmark_persist_directory = "benchmark_dataset_small_split"):    
    vectordb = Chroma(
        embedding_function=embeddings, 
        persist_directory=benchmark_persist_directory
    )
    #TODO get from config 
    dataset_path = "small_dataset_science_articles"
    # print(
    #             f"\n🔍 Searching for: \"{query}\" (similarity ≥ {min_similarity})")
    matches = search_similar(
                vectordb, query, min_similarity=min_similarity, topic=topic) #here I already search by topic
    if matches: 
        save_matches(matches)
        # Simple linear scaling
        n_articles = count_topic_articles(topic, dataset_path) #number of articles in topic (TODO double check if it's for one topic not the avg )
        freq_match_scaled = np.clip(len(matches) / n_articles, 0, 1) # Interpretation: 5 matches/topic = perfect score 

        return freq_match_scaled
    else: return 0.0

