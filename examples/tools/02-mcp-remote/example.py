import asyncio
import os

from dotenv import load_dotenv
from fastmcp import Client
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL") or ""
BASE_URL = os.getenv("BASE_URL") or "https://openrouter.ai/api/v1"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


async def main():
    if not TAVILY_API_KEY:
        print("Error: TAVILY_API_KEY not found in environment variables")
        print("Get your API key from: https://tavily.com")
        return

    server_config = {
        "command": "npx",
        "args": ["-y", "tavily-mcp@latest"],
        "env": {"TAVILY_API_KEY": TAVILY_API_KEY},
    }

    async with Client(server_config) as client:
        print("Available tools:")
        tools_list = await client.list_tools()
        for tool in tools_list.tools:
            print(f"  - {tool.name}: {tool.description}")

        tools = await load_mcp_tools(client.session)

        llm = ChatOpenAI(model=BASE_MODEL, base_url=BASE_URL, temperature=0)

        agent = create_agent(llm, tools)

        query = "Last 10 news about AI"

        response = await agent.ainvoke(query)

        print("Agent response:")
        for message in response["messages"]:
            if hasattr(message, "content") and message.content:
                print(f"\n{message.type}: {message.content[:500]}...")


if __name__ == "__main__":
    asyncio.run(main())
