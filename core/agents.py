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
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import init_chat_model
from os import path
import json

from .tools.storage_tools import (add_fact_tool, update_fact_tool,
                                  context_similarity_search_tool, fact_similarity_search_tool
                                  )



# Load prompts from YAML

PROMPT_FILE = path.join(path.dirname(__file__), "prompts.yaml")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)


model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0)
# Initialize LLM

# model= ChatOpenAI(
#         model="gpt-4o",
#         temperature=0.0
#     )

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

# Or change to llm = init_chat_model(model=config["chat_model"], 
                            #   model_provider=config["model_provider"],
                            #   temperature=0) 
def load_llm():
    cfg = load_config()

    provider = cfg.get("llm_provider", "ollama").lower()
    model_name = cfg.get("model_name", "gpt-oss:20b")
    temperature = cfg.get("temperature", 0.0)

    if provider == "ollama":
        print(f"Loading Ollama model: {model_name}")
        return ChatOllama(
            model=model_name,
            temperature=temperature
        )

    elif provider == "gpt":
        print(f"Loading OpenAI GPT model: {model_name}")
        # Load environment variables
        load_dotenv("chatbot.env")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


model = load_llm()

# Questions Manager Agent
questions_manager_agent = create_agent(
    model=model,
    tools=[context_similarity_search_tool, fact_similarity_search_tool],
    system_prompt=PROMPTS["questions_manager"],
    name="questions_manager_agent"
)


# Fact Extractor
class ExtractedFacts(BaseModel):
    facts: List[str] = Field(
        description="List of extracted facts in 'Subject: fact' format")


#TODO make a different models for config 
fact_extractor_llm = model.with_structured_output(ExtractedFacts)


def extract_facts_from_text(text: str) -> List[str]:
    """Extract multiple facts from a large text using the fact extractor LLM."""
    context = PROMPTS["fact_extractor"] + f"\n\nText to analyze:\n{text}"
    result = fact_extractor_llm.invoke(context)
    return result.facts


# Memory Manager Agent

memory_manager_tools = [add_fact_tool, update_fact_tool]

memory_manager_agent = create_agent(
    model = model,
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


supervisor_llm = model.with_structured_output(SupervisorResponse)


def route_user_request(user_input: str, chat_history: List[str] = None) -> SupervisorResponse:
    """Decide which agent should handle the user input."""
    context = PROMPTS["supervisor"] + f"\n\nUser message: {user_input}"
    if chat_history:
        context += f"\n\nChat history: {chat_history}"

    return supervisor_llm.invoke(context)
