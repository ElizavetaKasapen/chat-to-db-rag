"""
agent.py

Initializes all agents and provides helper functions
for fact extraction, memory management, and request routing.
"""

from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import yaml
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from os import path

from .tools.storage_tools import (add_fact_tool, update_fact_tool,
                                  context_similarity_search_tool, fact_similarity_search_tool
                                  )


# Load environment variables

load_dotenv("chatbot.env")


# Load prompts from YAML

PROMPT_FILE = path.join(path.dirname(__file__), "prompts.yaml")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)


# Initialize LLM

heavy_llm = ChatOllama(model="gpt-oss:20b")


# Questions Manager Agent

questions_manager_agent = create_agent(
    model=heavy_llm,
    tools=[context_similarity_search_tool, fact_similarity_search_tool],
    system_prompt=PROMPTS["questions_manager"],
    name="questions_manager_agent"
)


# Fact Extractor

class ExtractedFacts(BaseModel):
    facts: List[str] = Field(
        description="List of extracted facts in 'Subject: fact' format")


fact_extractor_llm = heavy_llm.with_structured_output(ExtractedFacts)


def extract_facts_from_text(text: str) -> List[str]:
    """Extract multiple facts from a large text using the fact extractor LLM."""
    context = PROMPTS["fact_extractor"] + f"\n\nText to analyze:\n{text}"
    result = fact_extractor_llm.invoke(context)
    return result.facts


# Memory Manager Agent

memory_manager_tools = [add_fact_tool, update_fact_tool]

memory_manager_agent = create_agent(
    model=heavy_llm,
    tools=memory_manager_tools,
    system_prompt=PROMPTS["memory_manager"],
    name="memory_manager_agent"
)


# Supervisor Agent

class SupervisorResponse(BaseModel):
    user_request: str = Field(
        description="Direct or reformulated user's request")
    called_agent: str = Field(
        description="The name of agent to call: questions_manager_agent or extract_facts_from_text")


supervisor_llm = heavy_llm.with_structured_output(SupervisorResponse)


def route_user_request(user_input: str, chat_history: List[str] = None) -> SupervisorResponse:
    """Decide which agent should handle the user input."""
    context = PROMPTS["supervisor"] + f"\n\nUser message: {user_input}"
    if chat_history:
        context += f"\n\nChat history: {chat_history}"

    return supervisor_llm.invoke(context)
