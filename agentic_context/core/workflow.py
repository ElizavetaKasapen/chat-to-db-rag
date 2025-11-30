# """
# workflow.py

# Handles user input processing:
# - Supervisor routing
# - Fact extraction
# - Memory management
# - Question answering
# """

# import logging
# from langchain_core.messages import ChatMessage, HumanMessage
# from core.tools.storage_tools import (
#     context_similarity_search_tool,
#     fact_similarity_search_tool,
#     get_facts_in_context_tool,
#     create_context_tool,
#     context_representation_builder
# )
# from core import agents as logic

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s"
# )


"""
workflow.py

Handles user input processing:
- Supervisor routing
- Fact extraction
- Memory management
- Question answering
"""

import logging
import json
from langchain_core.messages import HumanMessage
from core.tools.storage_tools import (
    context_similarity_search_tool,
    fact_similarity_search_tool,
    create_context_tool,
    context_representation_builder
)
from core import agents as logic

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s"
)

#TODO read it from json
CHAT_HISTORY_LIMIT = 5
SIMILAR_FACTS_LIMIT = 7


def _build_supervisor_context(user_input: str, chat_history: list) -> str:
    """
    Build context for Supervisor Agent.

    Args:
        user_input (str): User's message
        chat_history (list): Conversation history

    Returns:
        str: Formatted context string
    """
    context = logic.PROMPTS["supervisor"] + f"\n\nUser message: {user_input}"
    if chat_history:
        context += f"\n\nChat history: {chat_history[-CHAT_HISTORY_LIMIT:]}"
    return context


def _route_user_request(user_input: str, chat_history: list):
    """
    Route user request through Supervisor Agent.

    Args:
        user_input (str): User's message
        chat_history (list): Conversation history

    Returns:
        Routing decision object with 'user_request' and 'called_agent' fields
    """
    context = _build_supervisor_context(user_input, chat_history)
    routing = logic.supervisor_llm.invoke(context)
    logging.info(f"\n Agent: {routing.called_agent}\n")
    return routing


def _update_existing_context(context_id: str, facts: list[str]) -> str:
    """
    Update existing context with new facts.

    Args:
        context_id (str): ID of the context to update
        facts (list[str]): List of new facts to add

    Returns:
        str: Aggregated results of memory operations
    """
    answer = ""
    
    for fact in facts:
        # Find similar facts in the context
        existing_similar_facts = fact_similarity_search_tool.run({
            "context_id": context_id,
            "query": fact,
            "k": SIMILAR_FACTS_LIMIT
        })

        # Prepare request for Memory Manager
        request = {
            "messages": [
                HumanMessage(
                    content=f"\ncontext_id:{context_id}\nfact_text:{fact}\nExisting similar facts: {existing_similar_facts}"
                )
            ]
        }

        # Invoke Memory Manager Agent
        result = logic.memory_manager_agent.invoke(
            request, config={}, print_mode="debug"
        )
        
        for message in result["messages"]:
            message.pretty_print()
        
        answer += result['messages'][-1].content + "\n"
        print(f"FINAL ANSWER: {answer}")
    
    return answer


def _create_new_context(facts: list[str], representation: str) -> str:
    """
    Create new context with extracted facts.

    Args:
        facts (list[str]): List of facts to store
        representation (str): Vector representation of the context

    Returns:
        str: Confirmation message with context ID
    """
    result = create_context_tool.run({
        "fact_texts": facts,
        "representation": representation
    })
    context_id = result["context_id"]
    facts_info = result["facts"]
  
    return f"Created context '{context_id}' with {len(facts_info)} facts. Stored facts: {facts_info}\n"
    #return f"Created context '{context_id}' with {len(facts)} facts. Stored facts: {facts}"


def _process_factual_information(user_request: str) -> str:
    """
    Process factual information: extract facts and store in memory.

    Args:
        user_request (str): User's request containing factual information

    Returns:
        str: Result of memory operations or error message
    """
    # Extract facts from user input
    facts = logic.extract_facts_from_text(user_request)
    logging.info(f"Extracted facts: {facts}")

    if not facts:
        return "Sorry, I didn't find any facts or question. Could you repeat your request, please?"

    # Build context representation from facts
    input_context_text = context_representation_builder.build(facts)

    # Search for similar existing context
    similar_context = context_similarity_search_tool.run({
        "query": input_context_text
    })
    logging.info(f"Similar context found: {similar_context}")

    if similar_context:
        # Update existing context
        return _update_existing_context(similar_context, facts)
    else:
        # Create new context
        return _create_new_context(facts, input_context_text)


def _process_question(user_request: str) -> str:
    """
    Process user question through Questions Manager Agent.

    Args:
        user_request (str): User's question

    Returns:
        str: Answer based on stored knowledge
    """
    request = {"messages": [user_request]}
    result = logic.questions_manager_agent.invoke(request, config={})
    
    for message in result["messages"]:
        message.pretty_print()
    
    return result["messages"][-1].content


def process_user_input(user_input: str, chat_history: list) -> str:
    """
    Process a single user input message.

    Main workflow orchestrator that routes user input through appropriate
    processing pipeline based on Supervisor Agent decision.

    Args:
        user_input (str): The user's text input.
        chat_history (list): The last N messages from the conversation.

    Returns:
        str: The AI's response (memory operation results or question answer).
    """
   # logging.info(f"User input: {user_input}")

    # Route user request through Supervisor
    routing = _route_user_request(user_input, chat_history)

    # Process based on routing decision
    if routing.called_agent == "extract_facts_from_text":
        return _process_factual_information(routing.user_request)
    else:
        return _process_question(routing.user_request)







# def process_user_input(user_input: str, chat_history: list) -> str:
#     """
#     Process a single user input message.

#     Args:
#         user_input (str): The user's text input.
#         chat_history (list): The last N messages from the conversation.

#     Returns:
#         str: The AI's response.
#     """

#     logging.info(f"User input: {user_input}")

#     # Supervisor decides which agent to use
#     context = logic.PROMPTS["supervisor"] + f"\n\nUser message: {user_input}"
#     if chat_history:
#         # only last 5 messages
#         context += f"\n\nChat history: {chat_history[-5:]}"

#     routing = logic.supervisor_llm.invoke(context)
#     logging.info(f"\n Agent: {routing.called_agent}\n")
#         #f"\nRouting decision -> Request: {routing.user_request},\n Agent: {routing.called_agent}\n")

#     answer = ""

#     if routing.called_agent == "extract_facts_from_text":
#         # Extract facts from user's input
#         facts = logic.extract_facts_from_text(routing.user_request)
#         logging.info(f"Extracted facts: {facts}")

#         if facts:
#             # Build context embedding from facts
#             input_context_text = context_representation_builder.build(facts)

#             # Search for similar existing context
#             similar_context = context_similarity_search_tool.run(
#                 {"query": input_context_text})
#             print(f"similar_context: {similar_context}")
#             if similar_context:
#                 # Update memory for each fact
#                 for fact in facts:
#                     existing_similar_facts = fact_similarity_search_tool.run({ #use get_facts_in_context_tool to get all facts in the context
#                         "context_id": similar_context,
#                         "query": fact,
#                         "k": 5 #TODO create config?
#                     })

#                     request = {
#                         "messages": [
#                             HumanMessage(
#                                 content=f"\ncontext_id:{similar_context}\nFact:{fact}\nExisting similar facts: {existing_similar_facts}")
#                         ]
#                     }

#                     result = logic.memory_manager_agent.invoke(
#                         request, config={}, print_mode="debug")
#                     for message in result["messages"]:
#                         message.pretty_print()
#                     answer += result['messages'][-1].content + "\n"

#             else:
#                 # No similar context: create new context
#                 context_id = create_context_tool.run(
#                     {"fact_texts": facts, "representation": input_context_text})
#                 answer = f"Created context '{context_id}' with {len(facts)} facts. Stored facts: {facts}"
#         else:
#             answer = "Sorry, I didn't find any facts or question. Could you repeat your request, please?"

#     else:
#         # Route to question answering agent
#         request = {"messages": [routing.user_request]}
#         result = logic.questions_manager_agent.invoke(request, config={}) #TODO add chat story to the request 
#         for message in result["messages"]:
#             message.pretty_print()
#         answer = result["messages"][-1].content

#     return answer
