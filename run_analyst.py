"""Run the analyst agent and save the validated requirements artifact."""

import json
from pathlib import Path

from analyst_agent import run_analyst


PROJECT_DIR = Path(__file__).resolve().parent
BRIEF_FILE = PROJECT_DIR / "brief.txt"
OUTPUT_FILE = PROJECT_DIR / "artifacts" / "requirements.json"


def main() -> None:
    brief_text = BRIEF_FILE.read_text(encoding="utf-8")

    try:
        requirements = run_analyst(brief_text)
    except Exception as error:
        raise SystemExit(
            "Unable to generate requirements with Qwen. "
            "Make sure Ollama is running and qwen3:8b is installed. "
            f"Details: {error}"
        ) from error

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        json.dump(requirements, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"Validated requirements saved to {OUTPUT_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
