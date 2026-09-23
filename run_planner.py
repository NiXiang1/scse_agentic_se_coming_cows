"""Run the planner agent and save its validated plan."""

import json
from pathlib import Path

from planner_agent import run_planner


PROJECT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_DIR / "artifacts" / "requirements.json"
OUTPUT_FILE = PROJECT_DIR / "plan.json"


def main() -> None:
    try:
        with REQUIREMENTS_FILE.open(encoding="utf-8") as source:
            requirements = json.load(source)
        plan = run_planner(requirements)
        with OUTPUT_FILE.open("w", encoding="utf-8") as output:
            json.dump(plan, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except Exception as error:
        raise SystemExit(
            "Unable to generate plan with Qwen. Check artifacts/requirements.json, "
            "confirm Ollama is running, and confirm qwen3:8b is installed. "
            f"Details: {error}"
        ) from error

    print(f"Validated plan saved to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
