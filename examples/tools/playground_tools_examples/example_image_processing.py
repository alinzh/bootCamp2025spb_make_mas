import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from mas_lib.tools import describe_image

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL") or "gpt-4o-mini"


def main():
    llm = ChatOpenAI(model=BASE_MODEL, temperature=0)  # type: ignore

    agent = create_agent(llm, [describe_image])

    test_image = Path(__file__).parent.parent.parent / "data" / "test.png"

    query = f"Analyze the image at {test_image} and provide a detailed description of what you see."

    result = agent.invoke({"messages": [("user", query)]})

    print("Image Processing Agent Response:\n")
    for message in result["messages"]:
        if message.type == "ai" and hasattr(message, "content") and message.content:
            print(message.content)


if __name__ == "__main__":
    main()
