"""Run the developer agent and save its validated navigation code."""

import json
from pathlib import Path

from developer_agent import run_developer
from planner_agent import validate_plan


PROJECT_DIR = Path(__file__).resolve().parent
PLAN_FILE = PROJECT_DIR / "plan.json"
OUTPUT_FILE = PROJECT_DIR / "navigation_logic.py"


def main() -> None:
    try:
        with PLAN_FILE.open(encoding="utf-8") as source:
            plan = validate_plan(json.load(source))
        code = run_developer(plan)
        OUTPUT_FILE.write_text(code, encoding="utf-8")
    except Exception as error:
        raise SystemExit(
            "Unable to generate navigation code with Qwen. Check plan.json, "
            "confirm Ollama is running, and confirm qwen3:8b is installed. "
            f"Details: {error}"
        ) from error

    print(f"Validated navigation code saved to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
