# Week5Lab Crew

Welcome to the Week5Lab Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

Install LiteLLM proxy extras:

```bash
uv pip install 'litellm[proxy]'
```

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/week5lab/config/agents.yaml` to define your agents
- Modify `src/week5lab/config/tasks.yaml` to define your tasks
- Modify `src/week5lab/crew.py` to add your own logic, tools and specific args
- Modify `src/week5lab/main.py` to add custom inputs for your agents and tasks

## Running the Project

Primary command (from the project root, with this repo’s virtualenv active so `crewai` is the week5lab entry point):

```bash
crewai run 11
```

The number is the **1-based line index** in `data/test_review_subset.json` (JSONL): `1` is the first line, `11` is the eleventh row, and so on. That row’s `user_id` and `item_id` are sent to the crew. `crewai run` **must** include this id (no bare `crewai run`).

Same run via `uv` if `crewai` is not on your PATH:

```bash
uv run crewai run 11
# or
uv run run_crew 11
```

**Full test subset (every row):** one crew run per line; slow and many LLM calls. Use:

```bash
uv run test_eval
```

Outputs (generated files):

- `prediction_output.json` is the holder for the current prediction target output (the latest run only: predicted `stars` and `review`).
- `prediction_outputs.json` stores the full run records you want to track (e.g., `user_id`, `item_id`, `actual_stars`, and your predicted output).

**Optional:** pass explicit IDs as JSON instead of a line number:

```bash
crewai run '{"user_id":"<USER_ID>","item_id":"<ITEM_ID>"}'
```

On Windows PowerShell, prefer single quotes around the JSON, or escape double quotes so the process receives valid JSON with double-quoted keys.

## Understanding Your Crew

The week5lab Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the Week5Lab Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
