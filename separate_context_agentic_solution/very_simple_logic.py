from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from tools.storage_tools import (
    get_contexts_tool, get_facts_in_context_tool,
    create_context_tool, add_fact_tool, update_fact_tool, 
    context_similarity_search_tool,
    fact_similarity_search_tool)

from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, MessagesState, START
from typing import Optional, List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import ChatMessage

load_dotenv("chatbot.env")

# llm = ChatOpenAI(model="gpt-4o-mini")

llm = ChatOllama(model="gpt-oss:20b") # gpt-oss:20b. llama3.2 didn't extract the facts, 

questions_manager_prompt = ''' You are professional inforamational retriever. 
    Based on user's input you search for relevant context using context_similarity_search_tool, 
    after this you have to get context_id, which will be returned by this tool, and pass
    user's query to fact_similarity_search_tool. Based on retieved data, formulate friendly human-like
    answer to the user based only on retrieved data. If you don't know the answer, say that this information is not mentioned in your database.
    '''

questions_manager_agent = create_agent(
    model = llm,
    tools=[context_similarity_search_tool, fact_similarity_search_tool],
    system_prompt = questions_manager_prompt,
    name="questions_manager_agent",
)



#TODO add understanding of how I can understand what is important for the context 
#TODO add different types of context representation
fact_extractor_prompt = '''You are a professional fact extractor.

Your job is to analyze text and extract ALL individual facts as separate, atomic statements.

RULES:
1. Each fact should be self-contained and complete
2. Format each fact as "Subject: specific information about it"
3. Break compound statements into separate facts
4. Preserve all important details (dates, numbers, names, etc.)
5. Remove opinions, keep only factual statements

EXAMPLE INPUT:
"The Eiffel Tower was built in 1889 for the World's Fair. It's 330 meters tall and made of iron. 
The tower expands in summer heat by up to 15cm."

EXAMPLE OUTPUT:
- Eiffel Tower: was built in 1889
- Eiffel Tower: was built for the World's Fair
- Eiffel Tower: is 330 meters tall
- Eiffel Tower: is made of iron
- Eiffel Tower: expands by up to 15cm in summer heat

Extract ALL facts from the user's text.
'''
class ExtractedFacts(BaseModel):
    facts: List[str] = Field(description="List of extracted facts in 'Subject: fact' format")
    
fact_extractor_llm = llm.with_structured_output(ExtractedFacts)

def extract_facts_from_text(text: str) -> List[str]:
    """Extract multiple facts from a large text"""
    context = fact_extractor_prompt + f"\n\nText to analyze:\n{text}"
    result = fact_extractor_llm.invoke(context)
    return result.facts


# 1. Extract the core fact from user's message (format: "Object: fact about it")
#TODO maybe change get_facts_in_context_tool to fact_similarity_search_tool to keep llm context window ok
memory_manager_prompt = '''You are a professional memory manager. Your job is to store facts, NOT to chat with users.
You will receive MULTIPLE facts to store. For EACH fact:
PROCESS:
1. Search for similar context using get_contexts_tool
2. Handle based on result:

NO CONTEXT FOUND:
- Call create_context_tool with the fact
- Return ONLY: "Stored new context: <context_name>"

CONTEXT FOUND:
- Call get_facts_in_context_tool with context_id
- After retrieving similar facts from the context, evaluate whether the new fact is important.
  RULES:
    1. A fact is IMPORTANT if it helps understand, explain, prove, or clarify the main idea of the context.
    2. A fact is NOT IMPORTANT if it does not affect the meaning, reasoning, or conclusions.
    3. Consider relevance to key elements (names, dates, causes, effects) present in the similar facts.
  If the fact is NOT IMPORTANT, skip storing it and move to the next fact.
  If it IS IMPORTANT, continue with the following storage process: 
- Analyze the fact:

  a) DUPLICATE: If fact already exists
     Return ONLY: "Already stored: <existing_fact>"
  
  b) NEW INFO: If fact is new information in same context
     Call add_fact_tool(context_id, fact_text)
     Return ONLY: "Added to <context_id>: <new_fact>"
  
  c) CONTRADICTION or ENRICHMENT: If fact contradicts or adds details
     Call update_fact_tool(context_id, fact_id, updated_fact)
     Return ONLY: "Updated <context_id>: <old_fact> → <new_fact>"

Process EACH fact individually and completely before moving to the next.
Return summary: "Processed X facts: Y new, Z updated, W duplicates"

CRITICAL: 
- DO NOT explain or discuss facts
- DO NOT provide context or reasoning
- Just execute the tools and return the status message
- Keep responses under 20 words
'''

memory_manager_tools = [get_contexts_tool, get_facts_in_context_tool,
    create_context_tool, add_fact_tool, update_fact_tool]

memory_manager_agent = create_agent(
    model = llm,
    tools=memory_manager_tools,
    system_prompt = memory_manager_prompt,
    name="memory_manager_agent",
)



supervisor_prompt = '''You are the main manager of the system. 
Reformulate the request based on the past messages, so it can be understandable 
by itslef (for example, replace the pronouns with the actual name of the object). 
If it's already complete request, leave it as it is.
Analyze user's request and define is it just a question or an information that can be saved and used long-term.
Your ONLY job is to route requests to the appropriate agent.
If it's a question, pass the question to the questions_manager_agent.
If the user's request is a fact, that can be used in the future, pass it to extract_facts_from_text.
Return two parameters: user_request  - reformulated user's request if it was reformulated, otherwise 
just original user's request; called_agent - the agent you want to call: questions_manager_agent or extract_facts_from_text.
 ''' 



class SupervisorResponse(BaseModel):
    user_request: str = Field(description="Direct or reformulated (if needed) user's request")
    called_agent: str = Field(description="The name of agent to call: questions_manager_agent or extract_facts_from_text")

chat_history = []
supervisor_llm = llm.with_structured_output(SupervisorResponse)

user_input = "The Eiffel Tower can grow and shrink with the temperature."
context = supervisor_prompt + f"\n\nUser message: {user_input}"
if chat_history:
    context += f"\n\nChat history: {chat_history}"
    
routing = supervisor_llm.invoke(context)

print("\n Supervisor Decision:")
print(f"   Request: {routing.user_request}")
print(f"   Agent: {routing.called_agent}\n")


if routing.called_agent == "extract_facts_from_text":
    facts = extract_facts_from_text(routing.user_request)
    print(f"Extracted facts: {facts}")
    if facts:
        request = {
                "messages": [ ChatMessage(role="control", content="thinking"),
                    HumanMessage(
                    content=f"Facts:\n{facts}"
                )]
            }
        memory_answer = memory_manager_agent.invoke(request, config={}, print_mode="messages") # "updates"
        for message in memory_answer["messages"]: message.pretty_print()
        print(f"\n Final answer from memory_manager_agent: {memory_answer["messages"][-1].content}")
else:
    request = {"messages":[routing.user_request]}
    answer = questions_manager_agent.invoke(request, config={})
    for message in answer["messages"]: message.pretty_print()
    print(f"\n Final answer from questions_manager_agent: {answer["messages"][-1].content}")

