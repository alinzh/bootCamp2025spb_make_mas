import os
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent 
from dotenv import load_dotenv

load_dotenv(".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_PLANNER     = "google/gemini-2.0-flash-lite-001"
MODEL_SUPERVISOR  = "google/gemini-2.0-flash-lite-001"   # model supporting tool calling
MODEL_VALIDATOR   = "google/gemini-2.0-flash-lite-001"
MODEL_SUMMARIZER  = "google/gemini-2.0-flash-lite-001"


def make_openrouter_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        temperature=temperature,
    )


class State(TypedDict):
    messages: List[AnyMessage]
    plan: Optional[List[str]]
    draft: Optional[str]
    validated: Optional[bool]
    summary: Optional[str]


@tool
def web_search(query: str) -> str:
    """web-search"""
    return f"[mock] search results for: {query}"


@tool
def calc(expr: str) -> str:
    """calculator"""
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"calc error: {e}"


TOOLS = [web_search, calc]


planner_llm   = make_openrouter_llm(MODEL_PLANNER, temperature=0.0)
supervisor_llm= make_openrouter_llm(MODEL_SUPERVISOR, temperature=0.0)
validator_llm = make_openrouter_llm(MODEL_VALIDATOR, temperature=0.0)
summarizer_llm= make_openrouter_llm(MODEL_SUMMARIZER, temperature=0.3)


simple_system_prompt = "you are a helpful ai assistant. you must use the following tools to answer questions."
supervisor_agent_graph = create_agent(
    model=supervisor_llm,
    tools=TOOLS,
    system_prompt=simple_system_prompt
)


def planner_node(state: State) -> Dict[str, Any]:
    sys = SystemMessage(content="you are the planner. provide a brief plan (3–6 steps) to solve the task. do not solve it.")
    res = planner_llm.invoke([sys] + state["messages"])
    steps = [s.strip("- •").strip() for s in (res.content or "").split("\n") if s.strip()]
    return {"messages": state["messages"] + [res], "plan": steps[:8] or None}


def supervisor_node(state: State) -> Dict[str, Any]:
    # serialize langchain message objects to dicts with role, content
    def serialize_messages(messages):
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system"
        }
        serialized = []
        for m in messages:
            role = role_map.get(m.type, "user")
            serialized.append({"role": role, "content": m.content})
        return serialized


    messages = serialize_messages(state["messages"])
    result = supervisor_agent_graph.invoke({"messages": messages})


    draft = result["messages"][-1].content
    return {
        "messages": state["messages"] + result['messages'][2:],
        "draft": draft,
    }



def validator_node(state: State) -> Dict[str, Any]:
    draft = state.get("draft") or ""
    sys = SystemMessage(content="you are the validator. reply with json: {valid: bool, comment: str}.")
    ai = validator_llm.invoke([sys, HumanMessage(content=draft)])
    valid = "true" in (ai.content or "").lower()
    return {"messages": state["messages"] + [AIMessage(content=f"[validator] {ai.content}")],
            "validated": valid}


def summarizer_node(state: State) -> Dict[str, Any]:
    history=str(state["messages"])
    sys = SystemMessage(content=f"you are the summarizer. briefly summarize and provide the final answer. Text for summarization: {history}")
    ai = summarizer_llm.invoke([sys])
    return {"messages": state["messages"] + [AIMessage(content=f"[summary] {ai.content}")],
            "summary": ai.content}


graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("validator", validator_node)
graph.add_node("summarizer", summarizer_node)


graph.add_edge(START, "planner")
graph.add_edge("planner", "supervisor")
graph.add_edge("supervisor", "validator")
graph.add_edge("validator", "summarizer")
graph.add_edge("summarizer", END)


app = graph.compile()


if __name__ == "__main__":
    init: State = {
        "messages": [HumanMessage(content="calculate 21*2 and format the answer in one paragraph, use the calculator.")],
        "plan": None, "draft": None, "validated": None, "summary": None,
    }
    state = app.invoke(init)
    print("\n--- SUMMARY ---\n", state.get("summary"))
