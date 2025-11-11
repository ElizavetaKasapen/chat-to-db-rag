from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import AnyMessage
#from langchain.chat_models import ChatOpenAI
from tools import add_fact_to_store, get_all_contexts, get_facts_in_context, update_fact
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv("chatbot.env")

# ollama localhost localhost:11434 

llm = ChatOpenAI(model="gpt-4o-mini")



def questions_manager_prompt() -> list[AnyMessage]:
    system_msg = f''' You are professional inforamational retriever. 
    Based on user's input you search for relevant context using 
    
    '''

    return [{"role": "system", "content": system_msg}] + state["messages"]


questions_manager_agent = create_agent(
    model="openai:gpt-4.1",
    tools=[],
    prompt=prompt,
    name="questions_manager_agent",
)



supervisor_prompt=""


supervisor_graph = create_supervisor(
    model=init_chat_model("openai:gpt-4.1"),  # init_chat_model()
    agents=[
       
    ], 
    tools=[],
    prompt=supervisor_prompt,
    add_handoff_back_messages=True,
    allow_state_updates=True,
    output_mode="full_history",
)