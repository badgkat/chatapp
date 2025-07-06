"""
conductor.py
Central dispatcher that executes tool calls requested by the LLM.

A tool call is a dict like:
    { "tool": "web-search", "input": { "query": "..." } }

Each tool lives in tools/<name>.py and must implement:

    def run(input: dict) -> str        # returns plain-text result
"""
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

TOOLS_PATH = Path(__file__).parent / "tools"


class ToolError(RuntimeError):
    """Raised on any tool-dispatch failure."""


def _load_tool_module(name: str):
    # dash or space in names → underscore for module import
    mod_name = name.replace("-", "_").replace(" ", "_")
    return import_module(f"tools.{mod_name}")


def run_tool_call(call: Dict[str, Any]) -> str:
    if "tool" not in call:
        raise ToolError("Missing 'tool' key")
    tool_name = str(call["tool"])

    try:
        mod = _load_tool_module(tool_name)
    except ModuleNotFoundError:
        raise ToolError(f"No such tool: {tool_name}")

    if not hasattr(mod, "run"):
        raise ToolError(f"Tool {tool_name} lacks run()")

    return str(mod.run(call.get("input", {})))
