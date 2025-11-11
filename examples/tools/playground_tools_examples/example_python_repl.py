import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from mas_lib.tools import e2b_run_code

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL") or "gpt-4o-mini"


def main():
    llm = ChatOpenAI(model=BASE_MODEL, temperature=0)  # type: ignore

    agent = create_agent(llm, [e2b_run_code])

    query = "Calculate the sum of squares of numbers from 1 to 100, then find the average of those squares."

    result = agent.invoke({"messages": [("user", query)]})

    print("Python REPL Agent Response:\n")
    for message in result["messages"]:
        if message.type == "ai" and hasattr(message, "content") and message.content:
            print(message.content)


if __name__ == "__main__":
    main()
