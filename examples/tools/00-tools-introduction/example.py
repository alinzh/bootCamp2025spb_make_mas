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
    """Convert document to Markdown text format.

    Args:
        file_path: Path to document file (PDF, DOCX, XLSX, PPTX)

    Returns:
        Converted text content in Markdown format
    """
    try:
        expanded_path = os.path.expanduser(file_path)
        result = md.convert(expanded_path)
        return result.text_content
    except Exception as e:
        return f"ошибка конвертации: {str(e)}"


tools = [convert_to_markdown]

llm = ChatOpenAI(model=BASE_MODEL, base_url=BASE_URL, temperature=0)

graph = create_agent(llm, tools)


if __name__ == "__main__":
    test_pdf = Path(__file__).parent.parent.parent / "data" / "test.pdf"

    result = graph.invoke(
        f"Analyze the document at {test_pdf} and tell me what it contains"
    )

    print("Result:")
    for message in result["messages"]:
        if hasattr(message, "content") and message.content:
            print(f"{message.type}: {message.content}")
