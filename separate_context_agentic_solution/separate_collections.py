import uuid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIR = "./chroma_db_separate_context"

 
# Create Chroma collections
 
def get_contexts_store(persist_directory=PERSIST_DIR):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    contexts_store = Chroma(
        collection_name="contexts",
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    return contexts_store

def get_facts_store(context_id, persist_directory=PERSIST_DIR):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    facts_store = Chroma(
        collection_name=f"{context_id}",
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
    return facts_store

 
# Create a new context
 
def create_context(contexts_store, fact_text):
    context_id = str(uuid.uuid4())
    
    # Create facts store for this context
    facts_store = get_facts_store(context_id)
    
    # Add first fact to the facts collection
    # fact_id = str(uuid.uuid4())
    facts_store.add_texts(
        texts=[fact_text],
    )
    
    # Add context document (initially same as the first fact)
    contexts_store.add_texts(
        texts=[fact_text],
        metadatas=[{"context_id": context_id}],
        ids=[context_id]
    )
    
    return context_id


def update_context(context_id, contexts_store):
    # Update context text by concatenating all facts
    facts_store = get_facts_store(context_id)
    all_facts_docs = facts_store.get(include=["documents"])
    updated_context_text = " ".join(all_facts_docs["documents"])
    
    contexts_store.update_document(
        document_id=context_id,
        document=Document(
            updated_context_text,
            metadata={"context_id": context_id}
        )
    )

 
# Add a new fact and update context
 
def add_fact(contexts_store, context_id, fact_text):
    facts_store = get_facts_store(context_id)
    facts_store.add_texts(
        texts=[fact_text],
    )
    update_context(context_id, contexts_store)
    
    return f"Added fact '{fact_text}' to context {context_id}"


def update_fact(context_id, fact_id, updated_fact):
    facts_store = get_facts_store(context_id)
    facts_store.update_document(
        document_id=fact_id,
        document=Document(
            updated_fact,
            metadata={"updated": True}
        )
    )
    update_context(context_id, contexts_store)
    return f"Fact '{fact_id}' and its context were updated."



def delete_fact(contexts_store, context_id, fact_id):
    # Load the facts collection for this context
    facts_store = get_facts_store(context_id)
    
    # Delete the specific fact
    facts_store.delete(ids=[fact_id])
    print(f" Deleted fact '{fact_id}' from context '{context_id}'.")
    
    # Check if any facts remain
    remaining = facts_store.get(include=["ids"])
    
    if not remaining["ids"]:  # no facts left
        print(f" No more facts in context '{context_id}'. Deleting context and facts collection...")
        
        # Delete the context document
        contexts_store.delete(ids=[context_id])
        
        # Delete the entire facts collection from Chroma
        facts_store.delete_collection()
        
        return f"Context '{context_id}' and all related facts were deleted."
    
    else:
        # If facts remain, update the context text
        update_context(context_id, contexts_store)
        return f"Fact '{fact_id}' deleted and context '{context_id}' updated."


 
# Search context by query
 
# TODO add ability to retrieve a few context to compare if they really do represent it
def context_similarity_search(contexts_store, query, k=1): #TODO experiment with similarity_threshold
    results = contexts_store.similarity_search(query, k=k)
    print(f"results of context search: {results}")
    return results[0].metadata["context_id"]

 
# Search facts in a context
 
def fact_similarity_search(context_id, fact_text, top_k=5):
    facts_store = get_facts_store(context_id)
    results = facts_store.similarity_search(fact_text, k=top_k)
    print(f"fact_similarity_search: {results}")
    return results

 
# Get all contexts
 
def get_contexts(contexts_store):
    docs = contexts_store.get(include=["documents"])
    return [(doc_id, doc) for doc_id, doc in zip(docs["ids"], docs["documents"])]

 
# Get all facts in a context
 
def get_facts_in_context(context_id):
    facts_store = get_facts_store(context_id)
    docs = facts_store.get(include=["documents"])
    return [(doc_id, doc) for doc_id, doc in zip(docs["ids"], docs["documents"])]



################### USAGE ######################

 
# Initialize the contexts store
 
contexts_store = get_contexts_store()

 
# Create a new context with the first fact
 
# first_fact = "The Eiffel Tower is located in Paris, France."
# context_id = create_context(contexts_store, first_fact)
# print(f"Created context: {context_id}")

 
# Add more facts to the same context
 
# add_fact(contexts_store, context_id, "It was constructed in 1889 for the World's Fair.")
# add_fact(contexts_store, context_id, "The Eiffel Tower is 324 meters tall.")
# add_fact(contexts_store, context_id, "It is one of the most visited monuments in the world.")
# print("facts are added!")


query = "Where is the Eiffel Tower?"
found_context_id = context_similarity_search(contexts_store, query) #TODO add context comparison with llm

if found_context_id:
    print(f"\nFound relevant context: {found_context_id}")
else:
    print("\nNo relevant context found.")

#  
#  Search for similar facts inside that context
#  
if found_context_id:
    print(f"\nTop similar facts to: '{query}'")
    results = fact_similarity_search(found_context_id, query, top_k=3)
    for _, doc in enumerate(results, 1):
        print(f"{doc.id}. {doc.page_content}")


print(update_fact(found_context_id, '38542352-db3e-4a4f-bd47-02459f515de7', "The Eiffel Tower is  one of the most visited monuments in the world."))


# #  
#    Show all facts for a specific context
# #  
print(f"\nFacts in context {found_context_id}:")
for fid, fact in get_facts_in_context(found_context_id):
    print(f"- {fact}")



 
#   Show all stored contexts
 
print("\nAll contexts:")
for cid, text in get_contexts(contexts_store):
    print(f"- {cid[:8]}: {text}...")

# # ###### Second case 
# #  
# # # Create another context (different topic)
# #  
# second_fact = "The first human to walk on the Moon was Neil Armstrong in 1969."
# second_context_id = create_context(contexts_store, second_fact)

# print(f"Created another context: {second_context_id}")

# # Add more facts to this new context
# add_fact(contexts_store, second_context_id, "The Apollo 11 mission landed on the Moon on July 20, 1969.")
# add_fact(contexts_store, second_context_id, "Buzz Aldrin was the second person to walk on the Moon.")
# add_fact(contexts_store, second_context_id, "The command module pilot was Michael Collins.")

#  
# # Verify that both contexts exist
#  
# print("\nAll contexts in the database:")
# for cid, text in get_contexts(contexts_store):
#     print(f"- {cid[:8]}: {text[:80]}...")

#  
# #   Show facts of the new context
#  
# print(f"\nFacts in context {second_context_id}:")
# for fid, fact in get_facts_in_context(second_context_id):
#     print(f"- {fact}")