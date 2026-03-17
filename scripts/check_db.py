from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



persist_directory="./created_storages/demo"  #"./created_storages/concat_eval_claude"
embedding_model="all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
contexts_store = Chroma(
            collection_name="contexts",
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

# context_id = create_context_tool.run("The 2025 Nobel Peace Prize was awarded to María Corina Machado.")
# print("\n✅ Created context:", context_id)

# # # # Add a fact
# add_result = add_fact_tool.run({
#     "context_id": context_id,
#     "fact_text": "María Corina Machado received the prize for her tireless efforts to promote democratic rights and support a peaceful transition from dictatorship to democracy in Venezuela."
# })
# print("✅", add_result)

def get_facts_in_context(context_id: str):
        """Return all facts in a given context."""
        store = Chroma(
            collection_name=context_id,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        docs = store.get(include=["documents"])
        return list(zip(docs["ids"], docs["documents"]))

docs = contexts_store.get(include=["documents"])
contexts = list(zip(docs["ids"], docs["documents"])) 
# print(f"contexts: {contexts}")
def print_context():
    #("📂 All contexts stored:")
    for cid, text in contexts:
        print(f" \n - 📂 Context {cid} : {text}")
        facts = get_facts_in_context(cid)
        print(f"    \n📘 Facts inside '{cid}' context:")
        for fid, text in facts:
            print(f"  - {fid} : {text}")

    print()

print_context()