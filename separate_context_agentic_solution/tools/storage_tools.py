from langchain_core.tools import StructuredTool
from .storage_manager import ContextFactManager


manager = ContextFactManager()

# ------------------- Create Tools -------------------
create_context_tool = StructuredTool.from_function(
    func=manager.create_context,
    name="create_context",
    description="Create a new context with an initial fact."
)

add_fact_tool = StructuredTool.from_function(
    func=manager.add_fact,
    name="add_fact",
    description="Add a fact to an existing context."
)

update_fact_tool = StructuredTool.from_function(
    func=manager.update_fact,
    name="update_fact",
    description="Update an existing fact in a context and refresh the context."
)

delete_fact_tool = StructuredTool.from_function(
    func=manager.delete_fact,
    name="delete_fact",
    description="Delete a fact from a context; deletes context if last fact."
)

context_similarity_search_tool = StructuredTool.from_function(
    func=manager.context_similarity_search,
    name="context_similarity_search",
    description="Search among contexts and return the most relevant context_id."
)

fact_similarity_search_tool = StructuredTool.from_function(
    func=manager.fact_similarity_search,
    name="fact_similarity_search",
    description="Search for similar facts within a given context."
)

get_contexts_tool = StructuredTool.from_function(
    func=manager.get_contexts,
    name="get_contexts",
    description="Return list of (context_id, text) of all contexts."
)

get_facts_in_context_tool = StructuredTool.from_function(
    func=manager.get_facts_in_context,
    name="get_facts_in_context",
    description="Return list of (fact_id, text) for facts in a context."
)

# ------------------- Usage Examples -------------------
# # Create a context
# context_id = create_context_tool.run("The Eiffel Tower is located in Paris.")
# print("\n✅ Created context:", context_id)

# # Add a fact
# add_result = add_fact_tool.run({
#     "context_id": context_id,
#     "fact_text": "It was built in 1889."
# })
# print("✅", add_result)

# # Get facts in context
# facts = get_facts_in_context_tool.run({"context_id": context_id})
# print("\n📘 Facts in context before update:")
# for fid, text in facts:
#     print(f"  - {fid}... : {text}")

# # Update a fact
# if facts:
#     fact_id = facts[1][0]
#     update_result = update_fact_tool.run({
#         "context_id": context_id,
#         "fact_id": fact_id,
#         "updated_fact": "The Eiffel Tower was built in 1889."
#     })
#     print("\n✏️", update_result)

# # Show facts again after update
# facts_updated = get_facts_in_context_tool.run({"context_id": context_id})
# print("\n📘 Facts after update:")
# for fid, text in facts_updated:
#     print(f"  - {fid[:8]}... : {text}")

# # Context similarity search
# search_result = context_similarity_search_tool.run({"query": "Where is the Eiffel Tower?"})
# print("\n🔍 Context similarity search result:", search_result)

# # Fact similarity search
# fact_sim_result = fact_similarity_search_tool.run({
#     "context_id": context_id,
#     "fact_text": "When was the Eiffel Tower built?",
#     "top_k": 3
# })
# print("\n🔎 Fact similarity search results:", fact_sim_result)

# # Create another context for variety
# context2 = create_context_tool.run("The Earth orbits around the Sun.")
# add_fact_tool.run({"context_id": context2, "fact_text": "It takes 365 days for one complete orbit."})
# add_fact_tool.run({"context_id": context2, "fact_text": "The Earth is the third planet from the Sun."})
# print("\n🌍 Created second context with facts.\n")

# # Search for the most relevant context
# query = "Which monument in Paris is 324 meters tall?"
# best_context = context_similarity_search_tool.run({"query": query})
# print(f"🏗️ Most relevant context for query '{query}': {best_context}\n")

# # List all contexts
# contexts = get_contexts_tool.run({})
# print("📂 All contexts stored:")
# for cid, text in contexts:
#     print(f"  - {cid[:8]}... : {text[:80]}")
# print()

# # Delete a fact
# facts_to_delete = get_facts_in_context_tool.run({"context_id": context_id})
# if facts_to_delete:
#     fact_id = facts_to_delete[0][0]
#     del_result = delete_fact_tool.run({
#         "context_id": context_id,
#         "fact_id": fact_id
#     })
#     print("🗑️", del_result)

# # Show final state of contexts
# all_contexts = get_contexts_tool.run({})
# print("\n📚 Final contexts in store:")
# for cid, text in all_contexts:
#     print(f"  - {cid[:8]}... : {text[:100]}")