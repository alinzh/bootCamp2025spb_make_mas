import base64
import mimetypes
import os
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv
from e2b_code_interpreter import Execution, Sandbox
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from mas_lib.connectors.llm_connectors import get_llm_client
from langchain_tavily import TavilySearch
from pydantic import Field
from langchain_openai import ChatOpenAI


load_dotenv()

tavily_search = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

web_search = tavily_search


_sandbox_instances: dict[str, Sandbox] = {}


def get_sandbox(sandbox_id: Optional[str] = None) -> Sandbox:
    """Get or create a sandbox instance."""
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY environment variable not set")

    if sandbox_id and sandbox_id in _sandbox_instances:
        return _sandbox_instances[sandbox_id]

    try:
        if sandbox_id:
            sandbox = Sandbox.connect(sandbox_id, api_key=api_key)
        else:
            sandbox = Sandbox.create(api_key=api_key)

        _sandbox_instances[sandbox.sandbox_id] = sandbox
        return sandbox

    except Exception as e:
        raise RuntimeError(f"Failed to initialize sandbox: {str(e)}") from e


@tool
def e2b_run_code(
    code_block: Annotated[str, Field(description="The Python code to execute")],
    sandbox_id: Annotated[
        Optional[str],
        Field(
            description="Sandbox ID to run code in. If not provided, creates a new sandbox"
        ),
    ] = None,
) -> Execution | str:
    """Run Python code in E2B sandbox."""
    try:
        sandbox = get_sandbox(sandbox_id)
        result = sandbox.run_code(code_block)
        print("\n--- TOOL:  e2b---\n", code_block + '\n\n' + str(result))

        return result

    except Exception as e:
        return f"Code execution failed: {e}"


@tool
def describe_image(
    image_path: str, prompt: str = "Describe this image in detail"
) -> str:
    """Analyze and describe an image from a URL or LOCAL file path using vision model"""
    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
        model=os.getenv("VISION_MODEL", "mistralai/mistral-small-3.2-24b-instruct:free"),  # type: ignore
        temperature=0,
    )

    if image_path.startswith(("http://", "https://")):
        image_content = {"type": "image_url", "image_url": {"url": image_path}}
    else:
        path = Path(image_path).expanduser()
        if not path.exists():
            return f"Error: File not found: {image_path}"

        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"

        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        }

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            image_content,
        ]
    )

    try:
        response = model.invoke([message])
        print("\n--- TOOL:  describe_image---\n", response.content)

        return str(response.content)
    except Exception as e:
        return f"Error processing image: {e}"


@tool
def calc(expr: str) -> str:
    """calculator"""
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"calc error: {e}"