"""Analyst agent for converting a human brief into validated requirements."""

import json
import os
from typing import Any


MODEL_NAME = os.getenv("QWEN_MODEL", "qwen3:8b")
REQUIRED_KEYS = {
    "goal",
    "allowed_actions",
    "safe_stop",
    "avoid_obstacles",
}
ALLOWED_ACTIONS = ["FORWARD", "LEFT", "RIGHT", "STOP"]


def build_prompts(brief_text: str) -> tuple[str, str]:
    """Build the system and user prompts sent to Qwen."""
    system_prompt = """You are a software requirements analyst.
Your only task is to convert a human-language mobile robot brief into explicit
software requirements.

Return only one valid JSON object with exactly this structure:
{
  "goal": "string",
  "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
  "safe_stop": true,
  "avoid_obstacles": true
}

Rules:
- The goal must be a concise string describing the navigation objective.
- allowed_actions must contain exactly FORWARD, LEFT, RIGHT, and STOP in that order.
- safe_stop must be a JSON boolean derived from the brief.
- avoid_obstacles must be a JSON boolean derived from the brief.
- Do not add, remove, or rename any keys.
- Do not add any action other than the four allowed actions.
- Do not include Markdown, code fences, explanations, or text outside the JSON object.
"""
    user_prompt = f"Analyze this robot navigation brief:\n\n{brief_text}"
    return system_prompt, user_prompt


def ask_qwen(system_prompt: str, user_prompt: str) -> str:
    """Send prompts to Qwen through Ollama and return its response text."""
    from ollama import chat

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0},
    )

    if isinstance(response, dict):
        return response["message"]["content"]
    return response.message.content


def validate_requirements(result: Any) -> dict[str, Any]:
    """Validate Qwen's parsed output and return it when it is valid."""
    if not isinstance(result, dict):
        raise TypeError("Requirements must be a dictionary.")

    if set(result) != REQUIRED_KEYS:
        missing = sorted(REQUIRED_KEYS - set(result))
        extra = sorted(set(result) - REQUIRED_KEYS)
        raise ValueError(
            f"Requirements have incorrect keys. Missing: {missing}; extra: {extra}."
        )

    if not isinstance(result["goal"], str) or not result["goal"].strip():
        raise TypeError("goal must be a non-empty string.")

    actions = result["allowed_actions"]
    if not isinstance(actions, list):
        raise TypeError("allowed_actions must be a list.")
    if actions != ALLOWED_ACTIONS:
        raise ValueError(
            "allowed_actions must be exactly FORWARD, LEFT, RIGHT, and STOP."
        )

    if not isinstance(result["safe_stop"], bool):
        raise TypeError("safe_stop must be a boolean.")
    if not isinstance(result["avoid_obstacles"], bool):
        raise TypeError("avoid_obstacles must be a boolean.")

    return result


def run_analyst(brief_text: str) -> dict[str, Any]:
    """Ask Qwen to analyze the brief, parse its JSON, and validate the result."""
    if not isinstance(brief_text, str) or not brief_text.strip():
        raise ValueError("brief_text must be a non-empty string.")

    system_prompt, user_prompt = build_prompts(brief_text)
    response_text = ask_qwen(system_prompt, user_prompt)

    try:
        parsed_result = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen did not return valid JSON.") from error

    return validate_requirements(parsed_result)
