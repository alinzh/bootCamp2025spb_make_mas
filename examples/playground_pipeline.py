import os
from typing import Any, Dict

from connectors.llm_connectors import make_openrouter_llm
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from prompts import (
    planner_system_prompt,
    summary_system_prompt,
    supervisor_system_prompt,
    validator_system_prompt,
)
from state import State
from tools.playground_tools.tools import calc, web_search

load_dotenv(".env")

MODEL = os.environ.get("BASE_MODEL") or ""
MODEL_SUPERVISOR = os.environ.get("BASE_MODEL") or ""  # model supporting tool calling
TOOLS = [web_search, calc]

planner_llm = make_openrouter_llm(MODEL, temperature=0.0)
supervisor_llm = make_openrouter_llm(MODEL_SUPERVISOR, temperature=0.0)
validator_llm = make_openrouter_llm(MODEL, temperature=0.0)
summarizer_llm = make_openrouter_llm(MODEL, temperature=0.3)

supervisor_agent_graph = create_agent(
    model=supervisor_llm, tools=TOOLS, system_prompt=supervisor_system_prompt
)


def planner_node(state: State) -> Dict[str, Any]:
    sys = SystemMessage(content=planner_system_prompt)
    res = planner_llm.invoke([sys] + state["messages"])
    steps = [
        s.strip("- •").strip() for s in (res.content or "").split("\n") if s.strip()
    ]
    return {"messages": state["messages"] + [res], "plan": steps[:8] or None}


def supervisor_node(state: State) -> Dict[str, Any]:
    # serialize langchain message objects to dicts with role, content
    def serialize_messages(messages):
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        serialized = []
        for m in messages:
            role = role_map.get(m.type, "user")
            serialized.append({"role": role, "content": m.content})
        return serialized

    messages = serialize_messages(state["messages"])
    result = supervisor_agent_graph.invoke({"messages": messages})

    draft = result["messages"][-1].content
    return {
        "messages": state["messages"] + result["messages"][2:],
        "draft": draft,
    }


def validator_node(state: State) -> Dict[str, Any]:
    draft = state.get("draft") or ""
    sys = SystemMessage(content=validator_system_prompt)
    ai = validator_llm.invoke([sys, HumanMessage(content=draft)])
    valid = "true" in (ai.content or "").lower()
    return {
        "messages": state["messages"]
        + [AIMessage(content=f"[validator] {ai.content}")],
        "validated": valid,
    }


def summarizer_node(state: State) -> Dict[str, Any]:
    history = str(state["messages"])
    sys = SystemMessage(content=summary_system_prompt.format(history=history))
    ai = summarizer_llm.invoke([sys])
    return {
        "messages": state["messages"] + [AIMessage(content=f"[summary] {ai.content}")],
        "summary": ai.content,
    }


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
        "messages": [
            HumanMessage(
                content="calculate 21*2 and format the answer in one paragraph, use the calculator."
            )
        ],
        "plan": None,
        "draft": None,
        "validated": None,
        "summary": None,
    }
    state = app.invoke(init)
    print("\n--- SUMMARY ---\n", state.get("summary"))
