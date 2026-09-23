"""Planner agent for turning validated requirements into a navigation plan."""

import json
import os
from typing import Any


MODEL_NAME = os.getenv("QWEN_MODEL", "qwen3:8b")
PLAN_KEYS = {"strategy", "decisions", "stop_condition"}
DECISION_KEYS = {"condition", "action"}
ALLOWED_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP"}

SYSTEM_PROMPT = """You are the Planner Agent for a mobile robot.
Use only the validated requirements supplied by the user to decide how the
robot software should behave. Produce a deterministic, obstacle-safe plan.

Return only one valid JSON object with exactly this structure:
{
  "strategy": "non-empty navigation strategy",
  "decisions": [
    {"condition": "non-empty condition", "action": "FORWARD"}
  ],
  "stop_condition": "non-empty stop condition"
}

Rules:
- Include decisions for safely following a goal direction, choosing a safe
  fallback when that direction is blocked, and stopping when no direction is safe.
- Every action must be exactly FORWARD, LEFT, RIGHT, or STOP.
- Never plan movement into a blocked direction.
- Do not add, remove, or rename fields.
- Do not include Markdown, code fences, comments, or explanatory text.
"""


def validate_plan(data: Any) -> dict[str, Any]:
    """Validate a parsed planner response and return it unchanged."""
    if not isinstance(data, dict):
        raise TypeError("Plan must be a dictionary.")

    if set(data) != PLAN_KEYS:
        missing = sorted(PLAN_KEYS - set(data))
        extra = sorted(set(data) - PLAN_KEYS)
        raise ValueError(f"Plan has incorrect keys. Missing: {missing}; extra: {extra}.")

    if not isinstance(data["strategy"], str) or not data["strategy"].strip():
        raise TypeError("strategy must be a non-empty string.")
    if not isinstance(data["stop_condition"], str) or not data["stop_condition"].strip():
        raise TypeError("stop_condition must be a non-empty string.")

    decisions = data["decisions"]
    if not isinstance(decisions, list) or not decisions:
        raise TypeError("decisions must be a non-empty list.")

    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise TypeError(f"decisions[{index}] must be a dictionary.")
        if set(decision) != DECISION_KEYS:
            raise ValueError(
                f"decisions[{index}] must contain exactly condition and action."
            )
        if not isinstance(decision["condition"], str) or not decision["condition"].strip():
            raise TypeError(f"decisions[{index}].condition must be a non-empty string.")
        if decision["action"] not in ALLOWED_ACTIONS:
            raise ValueError(
                f"decisions[{index}].action must be FORWARD, LEFT, RIGHT, or STOP."
            )

    return data


def run_planner(requirement: dict[str, Any]) -> dict[str, Any]:
    """Ask Qwen for a plan based only on the supplied requirements."""
    if not isinstance(requirement, dict):
        raise TypeError("requirement must be a dictionary.")

    from ollama import chat

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Validated requirements:\n" + json.dumps(requirement),
            },
        ],
        format="json",
        think=False,
        options={"temperature": 0},
    )
    response_text = (
        response["message"]["content"]
        if isinstance(response, dict)
        else response.message.content
    )

    try:
        plan = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen did not return valid plan JSON.") from error

    return validate_plan(plan)
