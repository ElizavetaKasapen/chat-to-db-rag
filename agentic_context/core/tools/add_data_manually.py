from .storage_tools import (create_context_tool, add_fact_tool, 
                            get_facts_in_context_tool, update_fact_tool, 
                            context_similarity_search_tool,
                            fact_similarity_search_tool,
                            get_contexts_tool,
                            delete_context_tool
                            )


# # ------------------- Usage Examples -------------------
# # Create a context
# context_id = create_context_tool.run("The 2025 Nobel Peace Prize was awarded to María Corina Machado.")
# print("\n✅ Created context:", context_id)

# # # # Add a fact
# add_result = add_fact_tool.run({
#     "context_id": context_id,
#     "fact_text": "María Corina Machado received the prize for her tireless efforts to promote democratic rights and support a peaceful transition from dictatorship to democracy in Venezuela."
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
# search_result = context_similarity_search_tool.run({"query": "Who is María Corina Machado?"})
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

# # # Search for the most relevant context
# # query = "Which monument in Paris is 324 meters tall?"
# # best_context = context_similarity_search_tool.run({"query": query})
# # print(f"🏗️ Most relevant context for query '{query}': {best_context}\n")

# Delete context 
# delete_context_tool.run("e57983c6-3008-4e61-9ac0-1f46c829c11f")

# delete_context_tool.run("333a2bfd-20f3-48fb-9f56-aff21103007a")
# delete_context_tool.run("78ae01d4-689a-44cc-901e-756fe6add07c")
# delete_context_tool.run("c9549984-5bf2-43ac-8741-4b0db4097e57")

# List all contexts
def print_context():
    contexts = get_contexts_tool.run({})
    #("📂 All contexts stored:")
    for cid, text in contexts:
        print(f" \n - 📂 Context {cid} : {text}")
        facts = get_facts_in_context_tool.run({"context_id": cid})
        print(f"    \n📘 Facts inside '{cid}' context:")
        for fid, text in facts:
            print(f"  - {fid} : {text}")

    print()

# # # Delete a fact
# # facts_to_delete = get_facts_in_context_tool.run({"context_id": context_id})
# # if facts_to_delete:
# #     fact_id = facts_to_delete[0][0]
# #     del_result = delete_fact_tool.run({
# #         "context_id": context_id,
# #         "fact_id": fact_id
# #     })
# #     print("🗑️", del_result)

# # # Show final state of contexts
# all_contexts = get_contexts_tool.run({})
# print("\n📚 Final contexts in store:")
# for cid, text in all_contexts:
#     print(f"  - {cid[:8]}... : {text[:100]}")