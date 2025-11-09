import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL") or ""
BASE_URL = os.getenv("BASE_URL") or "https://openrouter.ai/api/v1"


async def main():
    server_script = Path(__file__).parent / "server.py"

    async with Client(str(server_script)) as client:
        print("Available tools:")
        tools_list = await client.list_tools()
        for tool in tools_list.tools:
            print(f"  - {tool.name}: {tool.description}")

        print("Available resources:")
        resources_list = await client.list_resources()
        for resource in resources_list.resources:
            print(f"  - {resource.uri}")

        tools = await load_mcp_tools(client.session)

        llm = ChatOpenAI(model=BASE_MODEL, base_url=BASE_URL, temperature=0)

        agent = create_agent(llm, tools)

        test_pdf = Path(__file__).parent.parent.parent / "data" / "test.pdf"

        response = await agent.ainvoke(
            f"Analyze the document at {test_pdf} and tell me what it contains"
        )

        print("Agent response:")
        for message in response["messages"]:
            if hasattr(message, "content") and message.content:
                print(f"\n{message.type}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
