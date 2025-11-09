from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class State(TypedDict, total=False):
    messages: List[BaseMessage]
    plan: Optional[List[str]]
    draft: Optional[str]
    validated: Optional[bool]
    summary: Optional[str]
    validation_fail_count: int
