# SCSE Requirements Engineering - coming_cows

This project converts the human-language robot navigation brief into explicit,
validated JSON requirements using Qwen through Ollama.

## Project structure

```text
scse_agentic_se_coming_cows/
├── artifacts/
│   └── requirements.json
├── analyst_agent.py
├── brief.txt
├── brief_to_req.py
├── robot_requirements.txt
├── run_analyst.py
└── requirements.txt
```

## Setup

Install Ollama and the model required by the course:

```bash
ollama run qwen3:8b
```

Create a virtual environment and install the Python dependency:

```bash
uv venv
uv pip install -r requirements.txt
```

## Run the initial experiment

```bash
uv run python brief_to_req.py
```

Run this several times and compare the generated `robot_requirements.txt` output
for consistency.

## Run the analyst agent

```bash
uv run python run_analyst.py
```

The validated result is saved to `artifacts/requirements.json`.

The course model defaults to `qwen3:8b`. For testing on another machine with a
different installed Qwen model, set `QWEN_MODEL` before running the scripts.
