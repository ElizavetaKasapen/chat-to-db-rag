from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import shutil
import numpy as np

def test_all_three_methods():
    temp_dir = tempfile.mkdtemp()

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}  # Change to "cpu" if no GPU
    )
    
    vectorstore = Chroma(
        collection_name="test",
        persist_directory=temp_dir,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Add test documents with clear similarity differences
    test_docs = [
        "The quick brown fox jumps over the lazy dog",      # ID: doc_0
        "The quick brown fox jumps over the lazy dog",      # ID: doc_1 (identical)
        "A fast brown fox leaps over a sleepy dog",         # ID: doc_2 (very similar)
        "Dogs and foxes are animals",                        # ID: doc_3 (somewhat related)
        "Python is a programming language",                  # ID: doc_4 (unrelated)
        "zxcvbnm qwertyuiop asdfghjkl"                      # ID: doc_5 (random)
    ]
    
    vectorstore.add_texts(test_docs, ids=[f"doc_{i}" for i in range(len(test_docs))])
    
    query = "The quick brown fox jumps over the lazy dog"
    query_embedding = embeddings.embed_query(query)
    
    print("=" * 80)
    print("TEST QUERY: 'The quick brown fox jumps over the lazy dog'")
    print("=" * 80)
    
    # METHOD 1: similarity_search_with_score
    print("\n" + "=" * 80)
    print("METHOD 1: similarity_search_with_score()")
    print("=" * 80)
    results1 = vectorstore.similarity_search_with_score(query, k=6)
    for i, (doc, score) in enumerate(results1):
        print(f"Rank {i+1}: Score = {score:.6f} | {doc.page_content}")
    
    # METHOD 2: similarity_search_with_relevance_scores
    print("\n" + "=" * 80)
    print("METHOD 2: similarity_search_with_relevance_scores()")
    print("=" * 80)
    results2 = vectorstore.similarity_search_with_relevance_scores(query, k=6)
    for i, (doc, score) in enumerate(results2):
        print(f"Rank {i+1}: Score = {score:.6f} | {doc.page_content}")
    
    # METHOD 3: similarity_search_by_vector_with_relevance_scores
    print("\n" + "=" * 80)
    print("METHOD 3: similarity_search_by_vector_with_relevance_scores()")
    print("=" * 80)
    results3 = vectorstore.similarity_search_by_vector_with_relevance_scores(
        query_embedding, k=6
    )
    for i, (doc, score) in enumerate(results3):
        print(f"Rank {i+1}: Score = {score:.6f} | {doc.page_content[:50]}")
    
    # ANALYSIS
    print("\n" + "=" * 80)
    print("ANALYSIS OF IDENTICAL DOCUMENT SCORES:")
    print("=" * 80)
    
    # Find scores for identical documents (should be doc_0 or doc_1)
    score1 = results1[0][1]
    score2 = results2[0][1]
    score3 = results3[0][1]
    
    print(f"\nIdentical document scores:")
    print(f"  Method 1 (with_score):                    {score1:.6f}")
    print(f"  Method 2 (with_relevance_scores):         {score2:.6f}")
    print(f"  Method 3 (by_vector_with_relevance...):   {score3:.6f}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    
    # Determine what each method returns
    if score1 < 0.1:
        print("✅ Method 1: Returns DISTANCE (0 = identical, higher = more different)")
        print("   → Use: if score < threshold")
        print("   → Suggested thresholds: 0.1-0.3 (strict), 0.4-0.6 (moderate)")
    else:
        print("✅ Method 1: Returns SIMILARITY (1 = identical, lower = more different)")
        print("   → Use: if score >= threshold")
        print("   → Suggested thresholds: 0.7-0.9 (strict), 0.5-0.7 (moderate)")
    
    if score2 > 0.9:
        print("\n✅ Method 2: Returns SIMILARITY (1 = identical, lower = more different)")
        print("   → Use: if score >= threshold")
        print("   → Suggested thresholds: 0.7-0.9 (strict), 0.5-0.7 (moderate)")
    else:
        print("\n✅ Method 2: Returns DISTANCE (0 = identical, higher = more different)")
        print("   → Use: if score < threshold")
        print("   → Suggested thresholds: 0.1-0.3 (strict), 0.4-0.6 (moderate)")
    
    if score3 > 0.9:
        print("\n✅ Method 3: Returns SIMILARITY (1 = identical, lower = more different)")
        print("   → Use: if score >= threshold")
        print("   → Suggested thresholds: 0.7-0.9 (strict), 0.5-0.7 (moderate)")
    else:
        print("\n✅ Method 3: Returns DISTANCE (0 = identical, higher = more different)")
        print("   → Use: if score < threshold")
        print("   → Suggested thresholds: 0.1-0.3 (strict), 0.4-0.6 (moderate)")
    
#         print("\n" + "=" * 80)
#         print("YOUR CODE SHOULD USE:")
#         print("=" * 80)
#         print("""
# def context_similarity_search(self, query, k=1, threshold=???):
#     if isinstance(query, np.ndarray):
#         # Method 3: similarity_search_by_vector_with_relevance_scores
#         results = self._contexts_store.similarity_search_by_vector_with_relevance_scores(
#             query.tolist(), k=k
#         )
#     else:
#         # Method 1: similarity_search_with_score
#         results = self._contexts_store.similarity_search_with_score(query, k=k)
    
#     if not results:
#         return None
    
#     doc, score = results[0]
    
#     # CHECK THE OUTPUT ABOVE TO DETERMINE:
#     # If Method 1 and Method 3 return DISTANCE: use (score < threshold)
#     # If Method 1 and Method 3 return SIMILARITY: use (score >= threshold)
#     # If they're DIFFERENT: you need to handle them separately!
#         """)
        

if __name__ == "__main__":
    test_all_three_methods()