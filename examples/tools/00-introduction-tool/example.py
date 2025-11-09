import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from markitdown import MarkItDown

load_dotenv()

md = MarkItDown()
BASE_MODEL = os.getenv("BASE_MODEL") or ""
BASE_URL = os.getenv("BASE_URL") or "https://openrouter.ai/api/v1"


@tool
def convert_to_markdown(file_path: str) -> str:
    try:
        expanded_path = os.path.expanduser(file_path)
        result = md.convert(expanded_path)
        return result.text_content
    except Exception as e:
        return f"ошибка конвертации: {str(e)}"


tools = [convert_to_markdown]

llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, temperature=0)  # type: ignore

graph = create_agent(llm, tools)


if __name__ == "__main__":
    test_pdf_path = Path(__file__).parent / "test.pdf"

    result = graph.invoke(f"Прочитай файл и расскажи что в нем {test_pdf_path}")

    print("\nResult:")
    for message in result["messages"]:
        if hasattr(message, "content") and message.content:
            print(f"{message.type}: {message.content}")
