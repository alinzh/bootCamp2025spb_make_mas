import os
from typing import Any, Dict, List

from mas_lib.connectors.llm_connectors import get_llm_client
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import END, START, StateGraph

from mas_lib.prompts import (
    planner_system_prompt,
    summary_system_prompt,
    supervisor_system_prompt,
    validator_system_prompt,
)
from mas_lib.tools import web_search, describe_image, e2b_run_code
from mas_lib.state import State

load_dotenv(".env")

MODEL = os.environ.get("BASE_MODEL")
MODEL_SUPERVISOR = os.environ.get("BASE_MODEL")  # model supporting tool calling
TOOLS = [web_search, describe_image, e2b_run_code]

planner_llm = get_llm_client(MODEL, temperature=0.1)
supervisor_llm = get_llm_client(MODEL_SUPERVISOR, temperature=0.1)
validator_llm = get_llm_client(MODEL, temperature=0.0)
summarizer_llm = get_llm_client(MODEL, temperature=0.1)

supervisor_agent_graph = create_agent(
    model=supervisor_llm, tools=TOOLS, system_prompt=supervisor_system_prompt, # debug=True
)

def _ensure_defaults(state: Dict[str, Any]) -> State:
    return {
        "messages": state.get("messages", []),
        "plan": state.get("plan"),
        "draft": state.get("draft"),
        "validated": state.get("validated"),
        "summary": state.get("summary"),
        "validation_fail_count": state.get("validation_fail_count", 0),
    }

def planner_node(state: State) -> Dict[str, Any]:
    sys = SystemMessage(content=planner_system_prompt)
    res = planner_llm.invoke([sys] + state["messages"])
    steps = [s.strip("- •").strip() for s in (res.content or "").split("\n") if s.strip()]

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [res]
    new_state["plan"] = steps[:8] or None
    
    print("\n--- PLANNER ---\n", steps[:8] or None)

    return _ensure_defaults(new_state)

def supervisor_node(state: State) -> Dict[str, Any]:
    def serialize_messages(messages: List[BaseMessage]):
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        serialized = []
        for m in messages:
            role = role_map.get(getattr(m, "type", "human"), "user")
            serialized.append({"role": role, "content": m.content})
        return serialized

    base_msgs = state["messages"][:2]
    result = supervisor_agent_graph.invoke({"messages": serialize_messages(base_msgs)})
    draft = result["messages"][-1].content
    appended = result["messages"][2:] if len(result["messages"]) > 2 else []

    new_state = dict(state)
    new_state["messages"] = state["messages"] + appended
    new_state["draft"] = draft
    
    print("\n--- SUPERVISOR ---\n", draft)
    
    return _ensure_defaults(new_state)

def validator_node(state: State) -> Dict[str, Any]:
    draft = state.get("draft") or ""
    sys = SystemMessage(content=validator_system_prompt)
    res = validator_llm.invoke([sys, HumanMessage(content=draft)])
    valid = "true" in (res.content or "").lower()

    count = state.get("validation_fail_count", 0)
    if not valid:
        count += 1

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [AIMessage(content=f"[validator] {res.content}")]
    new_state["validated"] = valid
    new_state["validation_fail_count"] = count
    
    print("\n--- VALIDATOR ---\n", res.content)

    return _ensure_defaults(new_state)

def summarizer_node(state: State) -> Dict[str, Any]:
    history = str(state["messages"])
    sys = SystemMessage(content=summary_system_prompt.format(history=history))
    res = summarizer_llm.invoke([sys])

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [AIMessage(content=f"[summary] {res.content}")]
    new_state["summary"] = res.content
    return _ensure_defaults(new_state)

graph = StateGraph(State)
graph.add_node("planner", planner_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("validator", validator_node)
graph.add_node("summarizer", summarizer_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "supervisor")
graph.add_edge("supervisor", "validator")

def validation_routing(state: State) -> bool:
    # True -> summarizer, False -> supervisor (повтор)
    validated = state.get("validated", False)
    fail_count = state.get("validation_fail_count", 0)
    return True if (validated or fail_count >= 2) else False

graph.add_conditional_edges("validator", validation_routing, {True: "summarizer", False: "supervisor"})
graph.add_edge("summarizer", END)

app = graph.compile()

if __name__ == "__main__":
    # query = 'Tell the story of ITMO University, using only confirmed information.'
    # query = 'Write code that will parse this page. What is about? Page: https://habr.com/ru/companies/spbifmo/articles/896868/'
    query = 'Describe the image https://cs1.htmlacademy.ru/blog/html/image-link/f7f41a6d72f8b2f2186fbe3614f87639.png'
    init: State = {
        "messages": [HumanMessage(content=query)],
        "plan": None,
        "draft": None,
        "validated": None,
        "summary": None,
        "validation_fail_count": 0,
    }
    state = app.invoke(init)
    print("\n--- SUMMARY ---\n", state.get("summary"))
