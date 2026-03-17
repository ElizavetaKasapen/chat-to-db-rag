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

#TODO also move to config?
one_lvl_directory = "one_lvl_facts"
benchmark_directory = "benchmark_dataset_small_split"
file_path = "generated_facts.json"
results_folder = "answer_estimation_results"
results_files = {
    "one_lvl": "one_lvl.txt",
    "benchmark": "benchmark.txt",
    "context_manager": "summary_context_manager.txt",
}


# TODO put all functions like this in separate folder like "utils" and call them
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)

def get_vectorstore(persist_directory):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        # TURNED ALL TO COSINE SIMILARITY
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(vectorstore.get(include=["documents"]))
    return vectorstore


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

with open("test_questions.json", "r") as f:
    questions = json.load(f)


# TODO compare: context_manager dataset (with mean, if it's the best one),
# dataset with the same facts, but without two-lvl storing,
# benchamark dataset (sentances separation)

for question in questions:
    print(f"Process {question}")
    start = time.time()
    one_lvl_response = standard_workflow(question, one_lvl_vs, k = 7)
    exec_time = time.time() - start
    results["one_lvl"].append((question, one_lvl_response, exec_time))


    start = time.time()
    benchmark_response = standard_workflow(question, benchmark_vs, k = 10) # k is more, cause split is too small
    exec_time = time.time() - start
    results["benchmark"].append((question, benchmark_response, exec_time))


    start = time.time()
    
    context_manager_response = process_user_input(question, chat_history = [])
    #print(f"context_manager_response: {context_manager_response}")
    exec_time = time.time() - start
    results["context_manager"].append((question, context_manager_response, exec_time))


avg_one_lvl = sum(item["execution_time"] for item in results["one_lvl"]) / len(results["one_lvl"])
avg_benchmark = sum(item["execution_time"] for item in results["benchmark"]) / len(results["benchmark"])
avg_context_manager = sum(item["execution_time"] for item in results["context_manager"]) / len(results["context_manager"])


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


os.makedirs(results_folder, exist_ok=True)

for key, filename in results_files.items():
    path = os.path.join(results_folder, filename)
    write_txt(path, results[key])
