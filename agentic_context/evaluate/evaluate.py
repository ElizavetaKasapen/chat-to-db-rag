"""
evaluation.py

Two-stage evaluation framework:
1. INGESTION: Process dataset through workflow.py to populate vector store
2. EVALUATION: Analyze the populated system against ground truth

Usage:
    python evaluation.py /path/to/dataset --output-dir ./results

The dataset should be organized as:
    dataset/
    ├── topic1/
    │   ├── article1.txt
    │   └── article2.txt
    └── topic2/
        └── article1.txt

This script will:
1. Process all articles through your workflow (stores to ChromaDB)
2. Track which facts came from which articles/topics
3. Evaluate context separation, fact importance, redundancy
4. Generate comprehensive reports
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from core.workflow import process_user_input
from core.tools.storage_tools import manager, context_representation_builder
from langchain_huggingface import HuggingFaceEmbeddings



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s : %(message)s"
)


class DatasetLoader:
    """Load articles from folder structure: dataset/topic/article.txt"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        
    def load(self) -> Tuple[List[Dict], List[str]]:
        """
        Load all articles with ground truth labels.
        
        Returns:
            articles: List of dicts with {text, topic, filename}
            ground_truth: List of topic labels (folder names)
        """
        articles = []
        ground_truth = []
        
        for topic_folder in self.dataset_path.iterdir():
            if not topic_folder.is_dir():
                continue
                
            topic_name = topic_folder.name
            logging.info(f"Loading topic: {topic_name}")
            
            for article_file in topic_folder.glob("*.txt"):
                try:
                    with open(article_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            article_data = {
                                "text": content,
                                "topic": topic_name,
                                "filename": article_file.name,
                                "full_path": str(article_file)
                            }
                            articles.append(article_data)
                            ground_truth.append(topic_name) # To know for sure the article from this topic is added
                            logging.info(f"  - Loaded: {article_file.name}")
                except Exception as e:
                    logging.error(f"Error loading {article_file}: {e}")
        
        logging.info(f"Total articles loaded: {len(articles)}")
        logging.info(f"Unique topics: {len(set(ground_truth))}")
        
        return articles, ground_truth


class IngestionTracker:
    """
    Track metadata during ingestion phase.
    
    Maps facts to their source articles and topics by analyzing
    stored facts after each article is processed.
    """
    
    def __init__(self):
        self.fact_metadata = {}  # fact_id -> {context_id, topic, article, fact_text}
        self.article_to_facts = defaultdict(list)  # article -> [fact_ids]
        self.context_snapshots = []  # Track contexts after each article
        
    def snapshot_after_article(self, article: Dict):
        """
        Take a snapshot of all contexts and facts after processing an article.
        Compare with previous snapshot to identify new facts.

        Only assign 'topic' and 'article' to facts that are newly added.
        Pre-existing facts keep their original topic.
        """
        current_contexts = manager.get_contexts()
        current_facts = {}

        # Build current facts without assigning topic yet
        for context_id, _ in current_contexts:
            facts = manager.get_facts_in_context(context_id)
            for fact_id, fact_text in facts:
                current_facts[fact_id] = {
                    "context_id": context_id,
                    "fact_text": fact_text
                    # Do NOT assign topic/article here
                }

        # Determine new facts compared to previous snapshot
        if self.context_snapshots:
            previous_fact_ids = set(self.context_snapshots[-1].keys())
            new_fact_ids = set(current_facts.keys()) - previous_fact_ids
        else:
            # First article: all facts are new
            new_fact_ids = set(current_facts.keys())

        # Assign topic/article only to new facts
        for fact_id in new_fact_ids:
            metadata = current_facts[fact_id]
            metadata.update({
                "topic": article["topic"],
                "article": article["filename"]
            })
            self.fact_metadata[fact_id] = metadata
            self.article_to_facts[article["filename"]].append(fact_id)

        # Keep previous facts intact
        for fact_id in current_facts:
            if fact_id not in new_fact_ids and fact_id not in self.fact_metadata:
                # Safety: if somehow fact is in store but not yet in fact_metadata
                self.fact_metadata[fact_id] = current_facts[fact_id]

        # Store snapshot for next comparison
        self.context_snapshots.append(current_facts)

        logging.info(
            f"Snapshot: {len(current_facts)} total facts, "
            f"{len(new_fact_ids)} new facts from this article"
        )
   
class ContextSeparationEvaluator:
    """Evaluate how well the system separates different topics into contexts."""
    
    def __init__(self, manager_instance):
        self.manager = manager_instance
        
    def evaluate(self, fact_metadata: Dict[str, Dict]) -> Dict[str, float]:
        """
        Evaluate context clustering quality.
        
        Args:
            fact_metadata: Dict mapping fact_id -> {context_id, topic, article, fact_text}
            
        Returns:
            Dictionary with clustering metrics
        """
        contexts = self.manager.get_contexts()
        
        if not contexts:
            logging.warning("No contexts found!")
            return {"error": "No contexts created"}
        
        if not fact_metadata:
            logging.warning("No fact metadata available!")
            return {"error": "No fact metadata"}
        
        # Build arrays for sklearn metrics
        predicted_labels = []
        true_labels = []
        
        for fact_id, metadata in fact_metadata.items():
            predicted_labels.append(metadata["context_id"])
            true_labels.append(metadata["topic"])
        
        if not predicted_labels:
            return {"error": "No facts with metadata"}
        
        # Calculate metrics
        purity = self._calculate_purity(fact_metadata)
        ari = self._calculate_ari(predicted_labels, true_labels)
        nmi = self._calculate_nmi(predicted_labels, true_labels)
        
        # Context distribution analysis
        context_distribution = self._analyze_context_distribution(fact_metadata)
        
        results = {
            "num_contexts_created": len(contexts),
            "num_ground_truth_topics": len(set(true_labels)),
            "num_facts_stored": len(fact_metadata),
            "purity": purity,
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "context_distribution": context_distribution,
            "perfect_separation": purity == 1.0 and ari == 1.0
        }
        
        return results
    
    def _calculate_purity(self, fact_metadata: Dict[str, Dict]) -> float:
        """Calculate purity: for each context, what % of facts are from dominant topic."""
        context_topics = defaultdict(list)
        
        for fact_id, metadata in fact_metadata.items():
            context_topics[metadata["context_id"]].append(metadata["topic"])
        
        total_correct = 0
        total_facts = 0
        
        for context_id, topics in context_topics.items():
            topic_counts = Counter(topics)
            most_common_count = topic_counts.most_common(1)[0][1]
            total_correct += most_common_count
            total_facts += len(topics)
        
        return total_correct / total_facts if total_facts > 0 else 0.0
    
    def _calculate_ari(self, predicted: List, true: List) -> float:
        """Adjusted Rand Index: measures clustering agreement."""
        try:
            return float(adjusted_rand_score(true, predicted))
        except Exception as e:
            logging.error(f"Error calculating ARI: {e}")
            return 0.0
    
    def _calculate_nmi(self, predicted: List, true: List) -> float:
        """Normalized Mutual Information: measures clustering quality."""
        try:
            return float(normalized_mutual_info_score(true, predicted))
        except Exception as e:
            logging.error(f"Error calculating NMI: {e}")
            return 0.0
    
    def _analyze_context_distribution(self, fact_metadata: Dict) -> Dict:
        """Analyze how facts are distributed across contexts."""
        context_topics = defaultdict(lambda: defaultdict(int))
        
        for fact_id, metadata in fact_metadata.items():
            context_topics[metadata["context_id"]][metadata["topic"]] += 1
        
        distribution = {}
        for context_id, topics in context_topics.items():
            total = sum(topics.values())
            distribution[context_id] = {
                "total_facts": total,
                "topic_distribution": dict(topics),
                "dominant_topic": max(topics.items(), key=lambda x: x[1])[0],
                "purity": max(topics.values()) / total
            }
        
        return distribution


class FactImportanceEvaluator:
    """Evaluate fact importance using multiple dimensions."""
    
    def __init__(self, manager_instance):
        self.manager = manager_instance
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"}
        )
        
    def evaluate_all_facts(self, fact_metadata: Dict[str, Dict]) -> pd.DataFrame:
        """Calculate importance scores for all stored facts."""
        results = []
        contexts = self.manager.get_contexts()
        
        for context_id, _ in contexts:
            facts_data = self.manager.get_facts_in_context(context_id)
            facts = [fact_text for _, fact_text in facts_data]
            
            for fact_id, fact_text in facts_data:
                meta = fact_metadata.get(fact_id, {})
                scores = self.calculate_importance(fact_text, context_id, facts)
                
                results.append({
                    "context_id": context_id,
                    "fact_id": fact_id,
                    "fact": fact_text[:100] + "..." if len(fact_text) > 100 else fact_text,
                    "fact_full": fact_text,
                    "topic": meta.get("topic", "unknown"),
                    "article": meta.get("article", "unknown"),
                    **scores
                })
        
        df = pd.DataFrame(results)
        return df
    
    def calculate_importance(
        self, 
        fact: str, 
        context_id: str,
        all_facts_in_context: List[str]
    ) -> Dict[str, float]:
        """Calculate multi-dimensional importance score for a single fact."""
        freq = self._frequency_score(fact, all_facts_in_context)
        cent = self._centrality_score(fact, context_id)
        idf = self._distinctiveness_score(fact)
        spec = self._specificity_score(fact)
        
        # Weighted combination
        importance = (
            0.3 * freq +
            0.4 * cent +
            0.2 * idf +
            0.1 * spec
        )
        
        return {
            "frequency": float(freq),
            "centrality": float(cent),
            "idf": float(idf),
            "specificity": float(spec),
            "importance": float(importance)
        }
    
    def _frequency_score(self, fact: str, all_facts: List[str]) -> float:
        """How many times similar content appears in the context."""
        if not all_facts or len(all_facts) == 1:
            return 0.0
        
        fact_embedding = self.embedding_fn.embed_query(fact)
        similar_count = 0
        
        for other_fact in all_facts:
            if other_fact == fact:
                continue
            other_embedding = self.embedding_fn.embed_query(other_fact)
            similarity = cosine_similarity(
                [fact_embedding], 
                [other_embedding]
            )[0][0]
            
            if similarity > 0.8:
                similar_count += 1
        
        return min(similar_count / (len(all_facts) - 1), 1.0)
    
    def _centrality_score(self, fact: str, context_id: str) -> float:
        """How close is this fact to the context's overall representation."""
        try:
            facts_store = self.manager._get_facts_store(context_id)
            context_repr = context_representation_builder.build(facts_store)
            
            if isinstance(context_repr, np.ndarray):
                fact_embedding = self.embedding_fn.embed_query(fact)
                similarity = cosine_similarity(
                    [fact_embedding],
                    [context_repr]
                )[0][0]
                return max(0.0, min(float(similarity), 1.0))
            
            elif isinstance(context_repr, str):
                fact_embedding = self.embedding_fn.embed_query(fact)
                context_embedding = self.embedding_fn.embed_query(context_repr)
                similarity = cosine_similarity(
                    [fact_embedding],
                    [context_embedding]
                )[0][0]
                return max(0.0, min(float(similarity), 1.0))
        except Exception as e:
            logging.warning(f"Error calculating centrality: {e}")
            return 0.5
        
        return 0.5
    
    def _distinctiveness_score(self, fact: str) -> float:
        """IDF-style: how unique is this fact across all contexts."""
        contexts = self.manager.get_contexts()
        total_contexts = len(contexts)
        
        if total_contexts == 0:
            return 0.0
        
        fact_embedding = self.embedding_fn.embed_query(fact)
        contexts_with_similar_fact = 0
        
        for context_id, _ in contexts:
            facts_data = self.manager.get_facts_in_context(context_id)
            
            for _, other_fact in facts_data:
                other_embedding = self.embedding_fn.embed_query(other_fact)
                similarity = cosine_similarity(
                    [fact_embedding],
                    [other_embedding]
                )[0][0]
                
                if similarity > 0.85:
                    contexts_with_similar_fact += 1
                    break
        
        if contexts_with_similar_fact == 0:
            return float(np.log(total_contexts + 1))
        
        idf = np.log(total_contexts / contexts_with_similar_fact)
        return max(0.0, float(idf))
    
    def _specificity_score(self, fact: str) -> float:
        """Length-based specificity: longer facts are usually more specific."""
        word_count = len(fact.split())
        return min(word_count / 15.0, 1.0)


class RedundancyEvaluator:
    """Measure redundancy: how similar are facts within the same context."""
    
    def __init__(self, manager_instance):
        self.manager = manager_instance
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"}
        )
    
    def evaluate(self) -> Dict[str, Any]:
        """Calculate redundancy metrics for all contexts."""
        contexts = self.manager.get_contexts()
        context_redundancies = {}
        all_similarities = []
        
        for context_id, _ in contexts:
            redundancy = self._calculate_context_redundancy(context_id)
            context_redundancies[context_id] = redundancy
            all_similarities.extend(redundancy["pairwise_similarities"])
        
        if all_similarities:
            results = {
                "per_context": context_redundancies,
                "overall_avg_similarity": float(np.mean(all_similarities)),
                "overall_max_similarity": float(np.max(all_similarities)),
                "overall_std_similarity": float(np.std(all_similarities)),
                "high_redundancy_pairs": sum(1 for s in all_similarities if s > 0.85),
                "medium_redundancy_pairs": sum(1 for s in all_similarities if 0.7 < s <= 0.85),
                "total_pairs_compared": len(all_similarities)
            }
        else:
            results = {"error": "No facts to compare"}
        
        return results
    
    def _calculate_context_redundancy(self, context_id: str) -> Dict[str, Any]:
        """Calculate redundancy within a single context."""
        facts_data = self.manager.get_facts_in_context(context_id)
        facts = [fact_text for _, fact_text in facts_data]
        
        if len(facts) < 2:
            return {
                "num_facts": len(facts),
                "avg_similarity": 0.0,
                "pairwise_similarities": []
            }
        
        embeddings = self.embedding_fn.embed_documents(facts)
        similarities = []
        
        for i, j in combinations(range(len(facts)), 2):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[j]]
            )[0][0]
            similarities.append(float(sim))
        
        return {
            "num_facts": len(facts),
            "avg_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
            "std_similarity": float(np.std(similarities)),
            "high_similarity_pairs": sum(1 for s in similarities if s > 0.85),
            "pairwise_similarities": similarities
        }


class SystemEvaluator:
    """Main evaluation orchestrator."""
    
    def __init__(self, dataset_path: str, output_dir: str = "./evaluation_results"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize evaluators
        self.context_eval = ContextSeparationEvaluator(manager)
        self.importance_eval = FactImportanceEvaluator(manager)
        self.redundancy_eval = RedundancyEvaluator(manager)
        
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        logging.info("=" * 60)
        logging.info("STAGE 1: INGESTION - Processing Dataset")
        logging.info("=" * 60)
        
        # Load dataset
        loader = DatasetLoader(self.dataset_path)
        articles, ground_truth = loader.load()
        
        if not articles:
            raise ValueError("No articles found in dataset!")
        
        # Process articles and track metadata
        tracker = IngestionTracker()
        processing_times = []
        
        total_start = time.time()
        
        for idx, article in enumerate(articles):
            article_start = time.time()
            
            logging.info(f"\n[{idx+1}/{len(articles)}] Processing: {article['topic']}/{article['filename']}")
            
            # Process through workflow
            response = process_user_input(article["text"], chat_history=[])
            
            # Take snapshot to identify new facts
            tracker.snapshot_after_article(article)
            
            article_elapsed = time.time() - article_start
            processing_times.append(article_elapsed)
            
            logging.info(f"  Time: {article_elapsed:.2f}s")
            logging.info(f"  Response: {response[:100]}...")
        
        total_elapsed = time.time() - total_start
        
        # Performance metrics
        perf_results = {
            "total_time": total_elapsed,
            "avg_time_per_article": float(np.mean(processing_times)),
            "min_time": float(np.min(processing_times)),
            "max_time": float(np.max(processing_times)),
            "std_time": float(np.std(processing_times)),
            "articles_per_second": len(articles) / total_elapsed,
            "total_facts_extracted": len(tracker.fact_metadata),
            "avg_facts_per_article": len(tracker.fact_metadata) / len(articles),
            "processing_times": [float(t) for t in processing_times]
        }
        
        logging.info(f"\nIngestion complete: {total_elapsed:.2f}s")
        logging.info(f"Total facts stored: {len(tracker.fact_metadata)}")
        
        # ===== STAGE 2: EVALUATION =====
        logging.info("\n" + "=" * 60)
        logging.info("STAGE 2: EVALUATION - Analyzing Stored Data")
        logging.info("=" * 60)
        
        # Context separation
        logging.info("\n--- Context Separation Evaluation ---")
        context_results = self.context_eval.evaluate(tracker.fact_metadata)
        logging.info(f"Contexts created: {context_results.get('num_contexts_created', 0)}")
        logging.info(f"Ground truth topics: {context_results.get('num_ground_truth_topics', 0)}")
        logging.info(f"Purity: {context_results.get('purity', 0):.3f}")
        logging.info(f"Adjusted Rand Index: {context_results.get('adjusted_rand_index', 0):.3f}")
        logging.info(f"Normalized Mutual Info: {context_results.get('normalized_mutual_info', 0):.3f}")
        
        # Fact importance
        logging.info("\n--- Fact Importance Evaluation ---")
        importance_df = self.importance_eval.evaluate_all_facts(tracker.fact_metadata)
        importance_stats = {
            "total_facts": len(importance_df),
            "avg_importance": float(importance_df["importance"].mean()),
            "std_importance": float(importance_df["importance"].std()),
            "avg_frequency": float(importance_df["frequency"].mean()),
            "avg_centrality": float(importance_df["centrality"].mean()),
            "avg_idf": float(importance_df["idf"].mean()),
            "avg_specificity": float(importance_df["specificity"].mean()),
            "high_importance_facts": int((importance_df["importance"] > 0.7).sum()),
            "low_importance_facts": int((importance_df["importance"] < 0.3).sum())
        }
        logging.info(f"Total facts analyzed: {importance_stats['total_facts']}")
        logging.info(f"Avg importance: {importance_stats['avg_importance']:.3f}")
        logging.info(f"High importance facts (>0.7): {importance_stats['high_importance_facts']}")
        
        # Redundancy
        logging.info("\n--- Redundancy Evaluation ---")
        redundancy_results = self.redundancy_eval.evaluate()
        if "overall_avg_similarity" in redundancy_results:
            logging.info(f"Avg similarity: {redundancy_results['overall_avg_similarity']:.3f}")
            logging.info(f"High redundancy pairs (>0.85): {redundancy_results['high_redundancy_pairs']}")
        
        # Compile results
        results = {
            "dataset_info": {
                "num_articles": len(articles),
                "num_topics": len(set(ground_truth)),
                "topics": list(set(ground_truth))
            },
            "performance": perf_results,
            "context_separation": context_results,
            "fact_importance": importance_stats,
            "redundancy": redundancy_results
        }
        
        # Save results
        self._save_results(results, importance_df, tracker.fact_metadata)
        
        logging.info("\n" + "=" * 60)
        logging.info("Evaluation Complete!")
        logging.info(f"Results saved to: {self.output_dir}")
        logging.info("=" * 60)
        
        return results
    
    def _save_results(
        self, 
        results: Dict, 
        importance_df: pd.DataFrame,
        fact_metadata: Dict
    ):
        """Save evaluation results to files."""
        
        # Save JSON summary
        json_path = self.output_dir / "evaluation_summary.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save importance details as CSV
        csv_path = self.output_dir / "fact_importance_details.csv"
        importance_df.to_csv(csv_path, index=False)
        
        # Save fact metadata
        metadata_path = self.output_dir / "fact_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(fact_metadata, f, indent=2)
        
        # Save human-readable report
        self._generate_report(results, self.output_dir / "evaluation_report.txt")
        
        logging.info(f"Saved summary: {json_path}")
        logging.info(f"Saved details: {csv_path}")
        logging.info(f"Saved metadata: {metadata_path}")
        logging.info(f"Saved report: {self.output_dir / 'evaluation_report.txt'}")
    
    def _generate_report(self, results: Dict, report_path: Path):
        """Generate human-readable report."""
        with open(report_path, 'w') as f:
            f.write("SYSTEM EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DATASET INFO\n")
            f.write("-" * 60 + "\n")
            f.write(f"Articles: {results['dataset_info']['num_articles']}\n")
            f.write(f"Topics: {results['dataset_info']['num_topics']}\n")
            f.write(f"Topic names: {', '.join(results['dataset_info']['topics'])}\n\n")
            
            f.write("PERFORMANCE\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total time: {results['performance']['total_time']:.2f}s\n")
            f.write(f"Avg per article: {results['performance']['avg_time_per_article']:.2f}s\n")
            f.write(f"Throughput: {results['performance']['articles_per_second']:.2f} articles/s\n")
            f.write(f"Total facts: {results['performance']['total_facts_extracted']}\n\n")
            
            f.write("CONTEXT SEPARATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Contexts created: {results['context_separation'].get('num_contexts_created', 0)}\n")
            f.write(f"Ground truth topics: {results['context_separation'].get('num_ground_truth_topics', 0)}\n")
            f.write(f"Facts stored: {results['context_separation'].get('num_facts_stored', 0)}\n")
            f.write(f"Purity: {results['context_separation'].get('purity', 0):.3f}\n")
            f.write(f"Adjusted Rand Index: {results['context_separation'].get('adjusted_rand_index', 0):.3f}\n")
            f.write(f"Normalized Mutual Info: {results['context_separation'].get('normalized_mutual_info', 0):.3f}\n")
            f.write(f"Perfect separation: {results['context_separation'].get('perfect_separation', False)}\n\n")
            
            f.write("FACT IMPORTANCE\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total facts: {results['fact_importance']['total_facts']}\n")
            f.write(f"Avg importance: {results['fact_importance']['avg_importance']:.3f}\n")
            f.write(f"High importance (>0.7): {results['fact_importance']['high_importance_facts']}\n")
            f.write(f"Low importance (<0.3): {results['fact_importance']['low_importance_facts']}\n")
            f.write(f"Avg frequency: {results['fact_importance']['avg_frequency']:.3f}\n")
            f.write(f"Avg centrality: {results['fact_importance']['avg_centrality']:.3f}\n")
            f.write(f"Avg IDF: {results['fact_importance']['avg_idf']:.3f}\n")
            f.write(f"Avg specificity: {results['fact_importance']['avg_specificity']:.3f}\n\n")
            
            if "overall_avg_similarity" in results['redundancy']:
                f.write("REDUNDANCY\n")
                f.write("-" * 60 + "\n")
                f.write(f"Avg similarity: {results['redundancy']['overall_avg_similarity']:.3f}\n")
                f.write(f"High redundancy pairs (>0.85): {results['redundancy']['high_redundancy_pairs']}\n")
                f.write(f"Medium redundancy pairs (0.7-0.85): {results['redundancy']['medium_redundancy_pairs']}\n")
                f.write(f"Total pairs compared: {results['redundancy']['total_pairs_compared']}\n\n")
            
            # Context distribution details
            if "context_distribution" in results['context_separation']:
                f.write("CONTEXT DISTRIBUTION DETAILS\n")
                f.write("-" * 60 + "\n")
                for ctx_id, dist in results['context_separation']['context_distribution'].items():
                    f.write(f"\nContext: {ctx_id[:8]}...\n")
                    f.write(f"  Total facts: {dist['total_facts']}\n")
                    f.write(f"  Dominant topic: {dist['dominant_topic']}\n")
                    f.write(f"  Context purity: {dist['purity']:.3f}\n")
                    f.write(f"  Topic breakdown: {dist['topic_distribution']}\n")


def main():
    """Run evaluation on dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate context-based memory management system"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset folder (contains topic subfolders with articles)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = SystemEvaluator(args.dataset_path, args.output_dir)
    results = evaluator.run_full_evaluation()
    
    print("\n✅ Evaluation complete!")
    print(f"📊 Results saved to: {args.output_dir}")
    print("\n📈 Quick Summary:")
    print(f"  - Contexts created: {results['context_separation'].get('num_contexts_created', 0)}")
    print(f"  - Facts stored: {results['context_separation'].get('num_facts_stored', 0)}")
    print(f"  - Purity: {results['context_separation'].get('purity', 0):.3f}")
    print(f"  - Processing time: {results['performance']['total_time']:.2f}s")


if __name__ == "__main__":
    main()