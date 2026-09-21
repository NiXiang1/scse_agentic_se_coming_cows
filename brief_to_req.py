import os
from pathlib import Path

from ollama import chat


PROJECT_DIR = Path(__file__).resolve().parent
BRIEF_FILE = PROJECT_DIR / "brief.txt"
OUTPUT_FILE = PROJECT_DIR / "robot_requirements.txt"
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen3:8b")


def main() -> None:
    brief_text = BRIEF_FILE.read_text(encoding="utf-8")
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": (
                    "Read the following human-language brief and produce clear, "
                    "explicit software requirements for the system:\n\n"
                    f"{brief_text}"
                ),
            }
        ],
        options={"temperature": 0},
    )

    content = (
        response["message"]["content"]
        if isinstance(response, dict)
        else response.message.content
    )
    OUTPUT_FILE.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Requirements saved to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
