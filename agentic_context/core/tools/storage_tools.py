"""
storage_tools.py

Defines LangChain StructuredTool wrappers around ContextFactManager.
"""

import json
from os import path
from langchain_core.tools import StructuredTool
from .storage_manager import ContextFactManager
from .context_representation import ContextRepresentation

# Load config
CONFIG_PATH = path.join(path.dirname(__file__), "storage_config.json")


def load_storage_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return json.load(f)


config = load_storage_config()

persist_directory = config["persist_directory"]
context_strategy = config["context_representation"]

context_representation_builder = ContextRepresentation(context_strategy)

manager = ContextFactManager(
    persist_directory=persist_directory,
    context_representation=context_representation_builder
)

# Tools
create_context_tool = StructuredTool.from_function(
    func=manager.create_context,
    name="create_context",
    description="Create a new context from a list of initial facts."
)

add_fact_tool = StructuredTool.from_function(
    func=manager.add_fact,
    name="add_fact",
    description="Add a new fact to an existing context."
)

update_fact_tool = StructuredTool.from_function(
    func=manager.update_fact,
    name="update_fact",
    description="Update an existing fact and regenerate the context."
)

delete_fact_tool = StructuredTool.from_function(
    func=manager.delete_fact,
    name="delete_fact",
    description="Delete a fact; deletes context if it becomes empty."
)

delete_context_tool = StructuredTool.from_function(
    func=manager.delete_context,
    name="delete_context",
    description="Delete a context and all its facts."
)

context_similarity_search_tool = StructuredTool.from_function(
    func=manager.context_similarity_search,
    name="context_similarity_search",
    description="Find the most similar context for a query."
)

fact_similarity_search_tool = StructuredTool.from_function(
    func=manager.fact_similarity_search,
    name="fact_similarity_search",
    description="Find similar facts inside a context."
)

get_contexts_tool = StructuredTool.from_function(
    func=manager.get_contexts,
    name="get_contexts",
    description="List all context IDs and their text representations."
)

get_facts_in_context_tool = StructuredTool.from_function(
    func=manager.get_facts_in_context,
    name="get_facts_in_context",
    description="List all facts belonging to a context."
)


get_context_embedding_tool = StructuredTool.from_function(
    func=manager.get_context_embedding,
    name="get_context_embedding",
    description="Return the embedding for a given context ID."
)


get_facts_embeddings_in_context_tool = StructuredTool.from_function(
    func=manager.get_facts_embeddings_in_context,
    name="get_facts_embeddings",
    description="Return the embedding for all facts according to a given context ID."
)
