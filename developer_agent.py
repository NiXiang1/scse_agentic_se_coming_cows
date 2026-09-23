"""Developer agent for generating validated robot navigation Python code."""

import ast
import json
import os
import re
from typing import Any


MODEL_NAME = os.getenv("QWEN_MODEL", "qwen3:8b")
ALLOWED_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP"}
REQUIRED_PARAMETERS = {
    "front_blocked",
    "left_blocked",
    "right_blocked",
    "goal_direction",
}

SYSTEM_PROMPT = """You are the Developer Agent for a mobile robot.
Generate Python source code that implements the navigation plan supplied by the user.

Output requirements:
- Output only valid Python source code. Do not use Markdown code fences and do not explain it.
- Define a function named navigate with exactly these parameters:
  front_blocked, left_blocked, right_blocked, goal_direction
- The three blocked parameters are booleans. goal_direction is one of
  "FORWARD", "LEFT", or "RIGHT".
- When the requested goal direction is safe, return that action.
- When it is blocked, choose a safe fallback in deterministic order:
  FORWARD, then LEFT, then RIGHT, skipping the blocked goal direction.
- When all three directions are blocked, return "STOP".
- Every return statement must directly return one of these string literals:
  "FORWARD", "LEFT", "RIGHT", or "STOP".
- Do not import modules, read files, use networking, or execute external commands.
"""


def _remove_code_fences(code_text: str) -> str:
    """Remove a single accidental Markdown code fence around model output."""
    text = code_text.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    return text + "\n"


def validate_code(code_text: str) -> str:
    """Validate syntax, navigation function shape, and possible return actions."""
    if not isinstance(code_text, str) or not code_text.strip():
        raise TypeError("Generated code must be a non-empty string.")

    code_text = _remove_code_fences(code_text)
    try:
        tree = ast.parse(code_text)
    except SyntaxError as error:
        raise ValueError(f"Qwen returned invalid Python: {error}") from error

    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise ValueError("Generated navigation code must not import modules.")

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    navigation_functions = [node for node in functions if node.name == "navigate"]
    if not navigation_functions:
        raise ValueError("Generated code must define a navigate function.")

    function = navigation_functions[0]
    parameters = {argument.arg for argument in function.args.args}
    if parameters != REQUIRED_PARAMETERS or len(function.args.args) != 4:
        raise ValueError(
            "navigate must take exactly front_blocked, left_blocked, "
            "right_blocked, and goal_direction."
        )

    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
    if not returns:
        raise ValueError("Generated navigation code must contain a return statement.")

    returned_actions = set()
    for node in returns:
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            raise ValueError("Every return must directly return an action string literal.")
        if node.value.value not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid returned action: {node.value.value!r}.")
        returned_actions.add(node.value.value)

    if "STOP" not in returned_actions:
        raise ValueError("navigate must be able to return STOP.")

    return code_text


def run_developer(plan: dict[str, Any]) -> str:
    """Ask Qwen to implement the supplied plan and validate its Python output."""
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary.")

    from ollama import chat

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Validated plan:\n" + json.dumps(plan)},
        ],
        think=False,
        options={"temperature": 0},
    )
    response_text = (
        response["message"]["content"]
        if isinstance(response, dict)
        else response.message.content
    )
    return validate_code(response_text)
