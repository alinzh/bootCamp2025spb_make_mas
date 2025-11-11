import os

from langchain_openai import ChatOpenAI

def get_llm_client(model: str = None, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.environ.get("BASE_MODEL"),
        base_url=os.environ.get("BASE_URL"),
        api_key=os.environ.get("LLM_API_KEY"),
        temperature=temperature,
    )
