# Run only after evaluate_storage
import os
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.workflow import process_user_input

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv("chatbot.env")
# TODO put all functions like this in separate folder like "utils" and call them
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0
)



embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)

one_lvl_directory = "one_lvl_facts"
benchmark_directory = "benchmark_dataset_small_split"

def get_vectorstore(persist_directory):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        # TURNED ALL TO COSINE SIMILARITY
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

file_path = "generated_facts.json"

def add_facts_to_one_lvl_vs(vectorstore):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)

            fact = entry["fact"]
            metadata = {
                "context": entry["context"],
                "topic": entry["topic"]
            }

            vectorstore.add_texts(
                texts=[fact],
                metadatas=[metadata]
            )

    # 4. Persist changes
    vectorstore.persist()
    print("Facts successfully added to Chroma!")


def standard_workflow(query, vectorsotre, k):
    facts = vectorsotre.similarity_search(query, k=k)
    prompt = f''' You are professional inforamational retriever. Based on retieved data, formulate friendly human-like
                answer to the user based only on retrieved data. If you don't know the answer, say that this information is not mentioned in your database.
                User's question: {query}. Retrieved data: {facts}'''
    response = llm.invoke(prompt)
    return response

def save_log(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# set up 
one_lvl_vs = get_vectorstore(one_lvl_directory)
if len(one_lvl_vs.get()) == 0:
    add_facts_to_one_lvl_vs(one_lvl_vs)

benchmark_vs = get_vectorstore(benchmark_directory)

results = {
    "one_lvl": [],
    "benchmark": [],
    "context_manager": []
}


# Asking questions
#TODO move to separate file
questions = ["What date was the 2021 Abu Dhabi Grand Prix held?",
             "Who won the 2025 Nobel Peace Prize?",
             "What was the name of Philippe Katerine's song at the Paris 2024 opening ceremony?",
             # TODO THIS IS INTERESTING CASE, check if it understands
             "Describe the sequence of events that led to Max Verstappen winning the 2021 championship, including the safety car procedure and tire strategies.",
             # Hallucination test
             "How many gold medals did Philippe Katerine win at the Paris Olympics?",
             "What did the White House say about Trump and the Nobel Prize?,"
             "What plans does Apple have for the iPhone Air 2?",
             "What happened to Michael Masi after Abu Dhabi 2021?",
             "What production issues are mentioned in the articles?",
             "What lap did Max Verstappen first pit for tires?",
             # Hallucination test
             "Did any Formula 1 drivers perform at the Paris Olympics opening ceremony?",
             ]

# TODO compare: context_manager dataset (with mean, if it's the best one),
# dataset with the same facts, but without two-lvl storing,
# benchamark dataset (sentances separation)

for question in questions:
    start = time.time()
    one_lvl_response = standard_workflow(question, one_lvl_vs, k = 7)
    exec_time = time.time() - start
    results["one_lvl"].append((question, one_lvl_response, exec_time))


    start = time.time()
    benchmark_response = standard_workflow(question, benchmark_vs, k = 10) # k is more cause split is too small
    exec_time = time.time() - start
    results["benchmark"].append((question, benchmark_response, exec_time))


    start = time.time()
    context_manager_response = process_user_input(question, chat_history = [])
    exec_time = time.time() - start
    results["context_manager"].append((question, context_manager_response, exec_time))


# avg_one_lvl = sum(item["execution_time"] for item in results["one_lvl"]) / len(results["one_lvl"])
# avg_benchmark = sum(item["execution_time"] for item in results["benchmark"]) / len(results["benchmark"])
# avg_context_manager = sum(item["execution_time"] for item in results["context_manager"]) / len(results["context_manager"])


results_folder = "answer_estimation_results"



def write_txt(path, entries):
    avg = sum(e[2] for e in entries) / len(entries)

    with open(path, "w", encoding="utf-8") as f:
        for i, (q, r, t) in enumerate(entries):
            f.write(f"--- ENTRY {i+1} ---\n")
            f.write(f"QUESTION:\n{q}\n\n")
            f.write(f"RESPONSE:\n{r}\n\n")
            f.write(f"EXECUTION TIME: {t:.6f} seconds\n")
            f.write("\n")

        f.write(f"=== AVERAGE EXECUTION TIME: {avg:.6f} seconds ===\n")

one_lvl_path = os.path.join(results_folder, 'one_lvl.txt')
benchmark_path = os.path.join(results_folder, 'benchmark.txt')
context_manager_path = os.path.join(results_folder, 'context_manager.txt')
os.makedirs(os.path.dirname(one_lvl_path), exist_ok=True)
os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
os.makedirs(os.path.dirname(context_manager_path), exist_ok=True)

write_txt(f"{one_lvl_path}", results["one_lvl"])
write_txt(f"{benchmark_path}", results["benchmark"])
write_txt(f"{context_manager_path}", results["context_manager"])