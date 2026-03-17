import os
import time
import json
import logging
from typing import Dict, Any
from collections import defaultdict
from itertools import combinations
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from core.workflow import process_user_input
from langchain_huggingface import HuggingFaceEmbeddings

from .calculate_frequency import calculate_frequency
from .get_mapping import extract_facts_dict
import tqdm
import numpy as np
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             fowlkes_mallows_score, v_measure_score, homogeneity_score,
                             completeness_score)
from collections import defaultdict
from core.tools.storage_tools import (get_context_embedding_tool, get_contexts_tool, 
                                      get_facts_embeddings_in_context_tool)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s : %(message)s"
)

output_file = "generated_facts.json"

def load_text_files(root_dir):
    for folder, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(folder, filename)
                folder_name = os.path.basename(folder)
                file_name = filename

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        yield full_path, folder_name, file_name, f.read()
                except Exception as e:
                    print(f"Skipping {full_path}: {e}")


def get_performance_metrics(total_elapsed, processing_times, processed_articles, n_facts):
    # Performance metrics
    return {
        "total_time": total_elapsed,
        "avg_time_per_article": float(np.mean(processing_times)),
        "min_time": float(np.min(processing_times)),
        "max_time": float(np.max(processing_times)),
        "std_time": float(np.std(processing_times)),
        "articles_per_second": processed_articles / total_elapsed,
        "avg_facts_per_article": n_facts / processed_articles,
        "processing_times": [float(t) for t in processing_times],
        "total_facts_extracted": n_facts
    }


def get_context_separation_metrics(all_facts):
    pred_labels, true_labels = get_labels(all_facts)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    fmi = fowlkes_mallows_score(true_labels, pred_labels)
    vmeasure = v_measure_score(true_labels, pred_labels)
    homogeneity = homogeneity_score(true_labels, pred_labels)
    completeness = completeness_score(true_labels, pred_labels)
    return {
        "num_created_contexts": len(set(pred_labels)),
        "num_ground_truth_topics": len(set(true_labels)),
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "fowlkes_mallows_score": fmi,
        "vmeasure": vmeasure,
        "homogeneity": homogeneity,
        "completeness": completeness
    }


def get_labels(facts):
    pred_labels = [f["context"] for f in facts]
    true_labels = [f["topic"] for f in facts]
    return pred_labels, true_labels


def iterate_facts_with_context(facts):
    facts_by_context = defaultdict(list)
    for f in facts:
        facts_by_context[f["context"]].append(f)

    for f in facts:
        yield (
            f["fact"],
            facts_by_context[f["context"]],
            f["context"],
            f["topic"]
        )


def centrality_score(fact: str, context_id: str, embeddings) -> float:
    """How close is this fact to the context's overall representation."""
    try:
        context_repr = get_context_embedding_tool.run(
            {"context_id": context_id})

        fact_embedding = embeddings.embed_query(fact)
        similarity = cosine_similarity(
            [fact_embedding],
            [context_repr]
        )[0][0]
        return max(0.0, min(float(similarity), 1.0))
    except Exception as e:
        logging.warning(f"Error calculating centrality: {e}")
        return 0.5


def specificity_score(fact: str, w_max = 10.0) -> float:
    """Length-based specificity: longer facts are usually more specific."""
    word_count = len(fact.split())
    return min(word_count / w_max, 1.0)


def distinctiveness_score(fact: str, embeddings) -> float:
    """IDF-style: how unique is this fact across all contexts."""
    contexts = get_contexts_tool.run({})
    total_contexts = len(contexts)
    if total_contexts == 0:
        return 0.0

    fact_emb = embeddings.embed_query(fact)
    contexts_with_similar_fact = 0

    for context_id, _ in contexts:
        facts_embeddings = get_facts_embeddings_in_context_tool.run({"context_id": context_id})
        if facts_embeddings.size == 0:
            print("No facts embeddings!")
            continue

        similarities = cosine_similarity([fact_emb], facts_embeddings)[0]

        if any(sim >= 0.85 for sim in similarities):
            contexts_with_similar_fact += 1

    # IDF with smoothing
    idf = np.log((total_contexts + 1) / (contexts_with_similar_fact + 1))
    return max(0.0, float(idf))


def get_facts_importance_metrics(all_facts):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )
    all_freq = np.array([])        
    all_cent = np.array([])        
    all_idf = np.array([])        
    all_spec = np.array([])        
    all_importance = np.array([]) 

    for fact, context_facts, context_id, topic in iterate_facts_with_context(all_facts):
        # print(f"Fact: {fact}")
        # print(f"Context: {context_id}")
        # print(f"topic: {topic}")
        # print("All facts:", [f["fact"] for f in context_facts])

        freq= calculate_frequency(topic, fact, embeddings)
        cent = centrality_score(fact, context_id, embeddings)
        idf = distinctiveness_score(fact, embeddings)
        spec = specificity_score(fact)

        # Weighted combination
        importance = (
            0.4 * freq +
            0.3 * cent +
            0.2 * idf +
            0.1 * spec
        )

        all_freq = np.append(all_freq, freq)
        all_cent = np.append(all_cent, cent)
        all_idf = np.append(all_idf, idf)
        all_spec = np.append(all_spec, spec)
        all_importance = np.append(all_importance, importance)


    return {  
            "total_facts": len(all_importance),
            "avg_importance": float(all_importance.mean()), 
            "std_importance": float(all_importance.std()),
            "avg_frequency": float(all_freq.mean()),
            "avg_centrality": float(all_cent.mean()),
            "avg_idf": float(all_idf.mean()),
            "avg_specificity": float(all_spec.mean()),
            "high_importance_facts": int(sum(1 for x in all_importance if x > 0.7)), 
            "low_importance_facts": int(sum(1 for x in all_importance if x < 0.3))
        }



#TODO put all help metrics in separate file
def calculate_context_redundancy(context_id: str) -> Dict[str, Any]:
    """Calculate redundancy within a single context."""
    facts_embeddings = get_facts_embeddings_in_context_tool.run({"context_id": context_id})
    num_facts = len(facts_embeddings)
    if num_facts < 2:
        return {
            "num_facts": num_facts,
            "avg_similarity": 0.0,
            "pairwise_similarities": []
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
    redundancy_ratio = high_similarity_pairs / total_pairs if total_pairs > 0 else 0
    
    return {
        "num_facts": num_facts,
        "avg_similarity": float(np.mean(similarities)),
        "max_similarity": float(np.max(similarities)),
        "min_similarity": float(np.min(similarities)),
        "std_similarity": float(np.std(similarities)),
        "high_similarity_pairs": high_similarity_pairs,
        "redundancy_ratio": redundancy_ratio,# size-independent measure for redundancy
        "pairwise_similarities": similarities
    }


def get_redundancy_metrics() -> Dict[str, Any]:
        """Calculate redundancy metrics for all contexts."""
        contexts = get_contexts_tool.run({})
        context_redundancies = {}
        all_similarities = []
        all_avg_similarities = []
        all_redundancy_ratio = []
        for context_id, _ in contexts:
            redundancy = calculate_context_redundancy(context_id)
            context_redundancies[context_id] = redundancy
            all_similarities.extend(redundancy["pairwise_similarities"]) # similarity of each pair of facts
            all_avg_similarities.append(redundancy["avg_similarity"]) # avg similarity inside one context 
            all_redundancy_ratio.append(redundancy["redundancy_ratio"])
        if all_avg_similarities:
            results = {
                "per_context": context_redundancies,
                # measurements for all contexts
                "overall_contex_avg_similarity": float(np.mean(all_avg_similarities)), # all_avg_similarities, cause if one context has more facts than another, it will have more weight
                "overall_contex_max_similarity": float(np.max(all_avg_similarities)),
                "overall_contex_std_similarity": float(np.std(all_avg_similarities)),
                "overall_contex_redundancy_ratio": float(np.mean(all_redundancy_ratio)),
                # general pairwise measurements
                "overall_max_similarity": float(np.max(all_similarities)),
                "overall_std_similarity": float(np.std(all_similarities)),
                "high_redundancy_pairs": sum(1 for s in all_similarities if s > 0.85),
                "medium_redundancy_pairs": sum(1 for s in all_similarities if 0.7 < s <= 0.85),
                "total_pairs_compared": len(all_similarities)
            }
        else:
            results = {"error": "No facts to compare"}
        
        return results


def main(dataset_path, saved_metrics_path = "all_metrics.json"):
    all_facts = []
    processing_times = []
    total_start = time.time()
    processed_articles = 0
    for path, folder_name, file_name, text in tqdm.tqdm(load_text_files(dataset_path)):
        logging.info(f"\nProcessing: {folder_name}/{file_name}")
        article_start = time.time()
        response = process_user_input(text, chat_history=[])
        extract_facts_dict(response, folder_name, output_file)

        processed_articles += 1
        article_elapsed = time.time() - article_start
        processing_times.append(article_elapsed)

        logging.info(f"  Time: {article_elapsed:.2f}s")
        logging.info(f"  Response: {response[:100]}...")

    total_elapsed = time.time() - total_start
    logging.info(f"\nIngestion complete: {total_elapsed:.2f}s")
    with open(output_file, "r", encoding="utf-8") as f:
        all_facts = json.load(f)
    print(f"Extracted_facts:{all_facts}")
    n_facts = len(all_facts)
    logging.info(f"Total facts stored: {n_facts}")
    # for fact_entry in all_facts:  # TODO delete later
    #     print(fact_entry)

    # performance metrics
    performance_metrics = get_performance_metrics(
        total_elapsed, processing_times, processed_articles, n_facts)
    print(f"performance_metrics: {performance_metrics}")
    # context separation metrics
    context_separation_metrics = get_context_separation_metrics(all_facts)
    print(f"context_separation_metrics: {context_separation_metrics}")
    # importance metrics
    importance_metrics = get_facts_importance_metrics(all_facts) #it's just easier to me to pass the facts like this, cause I already creted the functions, in the future, I will take them from the vs
    print(f"importance_metrics: {importance_metrics}")
    # redundancy metrics
    redundancy_metrics = get_redundancy_metrics()
    print(f"redundancy_metrics: {redundancy_metrics}")
    # save all metrics to json
    all_metrics = {
        "performance_metrics": performance_metrics,
        "context_separation_metrics": context_separation_metrics,
        "importance_metrics": importance_metrics,
        "redundancy_metrics": redundancy_metrics
    }
    with open(saved_metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)


#TODO change this call (maybe do it as termonal args)
# "small_dataset_science_articles"  "evaluation_results\\concat_metrics.json"
main("small_dataset_science_articles","evaluation_results\\new_mean_objects_metrics.json") 