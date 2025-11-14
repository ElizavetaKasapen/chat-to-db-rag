from langchain_core.tools import StructuredTool
from .storage_manager import ContextFactManager
import json
from os import path


config_path = path.join("tools", "storage_config.json")
def load_storage_config(path=config_path):
    with open(path, "r") as f:
        return json.load(f)

config = load_storage_config()
#TODO some checks
manager = ContextFactManager(
    persist_directory=config["persist_directory"],
    context_representation=config["context_representation"]
)

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

delete_context_tool = StructuredTool.from_function(
    func=manager.delete_context,
    name="delete_context",
    description= "Deletes an entire context and all its associated facts."
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
