import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv("chatbot.env")

class ContextRepresentation:
    def __init__(self, context_type: str):
        self.context_type = context_type
        #self.llm = ChatOpenAI(temperature=0)
        self.llm = ChatOllama(model="gpt-oss:20b")

        self._strategies = {
            "mean": self._mean,
            "main_objects": self._main_objects,
            "summary": self._summary,
        }

    def build(self, facts_store):
        """Main API: automatically retrieves docs from the store."""
        strategy = self._strategies.get(self.context_type)

        if strategy is None:
            return self._default(facts_store)

        return strategy(facts_store)


    def _mean(self, facts_store):
        """Compute mean embedding using stored embeddings."""
        data = facts_store.get(include=["embeddings"])

        vectors = data["embeddings"]
        print(f"Context vectors: {vectors}")
        if  len(vectors) > 0:
            return np.mean(vectors, axis=0)
        
        return None

    def _main_objects(self, facts_store):
        """Extract key entities from all stored docs."""
        data = facts_store.get(include=["documents"])
        docs = data["documents"]
        text = " ".join(docs)

        prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Extract the main entities (people, places, organizations, objects, "
                "and times) from the following text:\n\n{text}\n\n"
                "Return them as a concise comma-separated list."
            ),
        )
        objects = self.llm.invoke(prompt.format(text=text))
        return objects.content

    def _summary(self, facts_store):
        """Summarize all stored docs."""
        data = facts_store.get(include=["documents"])
        docs = data["documents"]
        text = " ".join(docs)

        prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Summarize the following text concisely while preserving key information:\n\n{text}"
            ),
        )
        summary = self.llm.invoke(prompt.format(text=text))
        return summary.content

    def _default(self, facts_store):
        """Fallback: just return concatenated docs."""
        data = facts_store.get(include=["documents"])
        return " ".join(data["documents"])
