"""
workflow.py

Handles user input processing:
- Supervisor routing
- Fact extraction
- Memory management
- Question answering
"""

import logging
from langchain_core.messages import ChatMessage, HumanMessage
from core.tools.storage_tools import (
    context_similarity_search_tool,
    fact_similarity_search_tool,
    get_facts_in_context_tool,
    create_context_tool,
    context_representation_builder
)
import asyncio
from core import agents as logic

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s"
)



def process_user_input(user_input: str, chat_history: list) -> str:
    """
    Process a single user input message.

    Args:
        user_input (str): The user's text input.
        chat_history (list): The last N messages from the conversation.

    Returns:
        str: The AI's response.
    """

    logging.info(f"User input: {user_input}")

    # Supervisor decides which agent to use
    context = logic.PROMPTS["supervisor"] + f"\n\nUser message: {user_input}"
    if chat_history:
        # only last 5 messages
        context += f"\n\nChat history: {chat_history[-5:]}"

    routing = logic.supervisor_llm.invoke(context)
    logging.info(
        f"\nRouting decision -> Request: {routing.user_request},\n Agent: {routing.called_agent}\n")

    answer = ""

    if routing.called_agent == "extract_facts_from_text":
        # Extract facts from user's input
        facts = logic.extract_facts_from_text(routing.user_request)
        logging.info(f"Extracted facts: {facts}")

        if facts:
            # Build context embedding from facts
            input_context_text = context_representation_builder.build(facts)

            # Search for similar existing context
            similar_context = context_similarity_search_tool.run(
                {"query": input_context_text})
            print(f"similar_context: {similar_context}")
            if similar_context:
                # Update memory for each fact
                for fact in facts:
                    existing_similar_facts = fact_similarity_search_tool.run({ #use get_facts_in_context_tool to get all facts in the context
                        "context_id": similar_context,
                        "query": fact,
                        "k": 5 #TODO create config?
                    })

                    request = {
                        "messages": [
                            ChatMessage(role="control", content="thinking"),
                            HumanMessage(
                                content=f"\ncontext_id:{similar_context}\nFact:{fact}\nExisting similar facts: {existing_similar_facts}")
                        ]
                    }

                    result = logic.memory_manager_agent.invoke(
                        request, config={}, print_mode="debug")
                    for message in result["messages"]:
                        message.pretty_print()
                    answer += result['messages'][-1].content + "\n"

            else:
                # No similar context: create new context
                context_id = create_context_tool.run(
                    {"fact_texts": facts, "representation": input_context_text})
                answer = f"Created context '{context_id}' with {len(facts)} facts. Stored facts: {facts}"
        else:
            answer = "Sorry, I didn't find any facts or question. Could you repeat your request, please?"

    else:
        # Route to question answering agent
        request = {"messages": [routing.user_request]}
        result = logic.questions_manager_agent.invoke(request, config={})
        for message in result["messages"]:
            message.pretty_print()
        answer = result["messages"][-1].content

    return answer
