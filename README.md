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

**LLM provider.** The crew now defaults to **NVIDIA NIM** (OpenAI-compatible
endpoint at `https://integrate.api.nvidia.com/v1`) running
`minimaxai/minimax-m2.7`. Put the following into your `.env`:

```bash
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL_NAME=minimaxai/minimax-m2.7
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
```

To fall back to Ollama instead, set `LLM_PROVIDER=ollama` and `MODEL=ollama/<tag>`
(e.g. `ollama/phi3`). See `.env.example` for the full list of knobs.

- Modify `src/week5lab/config/agents.yaml` to define your agents
- Modify `src/week5lab/config/tasks.yaml` to define your tasks
- Modify `src/week5lab/crew.py` to add your own logic, tools and specific args
- Modify `src/week5lab/main.py` to add custom inputs for your agents and tasks

## Running the Project

Primary command (from the project root, with this repo’s virtualenv active):

```bash
crewai run 11
```

The number is the **1-based line index** in `data/test_review_subset.json` (JSONL): `1` is the first line, `11` is the eleventh row, and so on. That row’s `user_id` and `item_id` are sent to the crew. `crewai run` **must** include one of: a row index, a JSON payload, or `--user_id` / `--item_id` flags (see below).

**Other forms** (same `crewai run` entry point):

```bash
crewai run '{"user_id":"<USER_ID>","item_id":"<ITEM_ID>"}'
crewai run --user_id "<USER_ID>" --item_id "<ITEM_ID>"
crewai run --userid:"<USER_ID>" --itemid="<ITEM_ID>"
```

### Choosing a collaboration pattern (`--crew`)

The project ships **three** Crew variants from the Week 5 lab, all driven by
the same CLI:

| Mode            | Process                | Pattern                                           |
|-----------------|------------------------|---------------------------------------------------|
| `sequential`    | `Process.sequential`   | Pattern 1 — Research → Write → Edit (default)     |
| `collab`        | `Process.sequential`   | Pattern 2 — Collaborative single task             |
| `hierarchical`  | `Process.hierarchical` | Pattern 3 — Manager agent decomposes & delegates  |

Pick one with `--crew`:

```bash
uv run run_crew --crew collab 3
uv run run_crew --crew hierarchical 3
crewai run --crew collab 3
```

You can also set the env var `WEEK5LAB1_CREW_MODE=collab` to change the
default without passing `--crew`.

All three crews enable **memory** (`memory=True` on the Crew) and
**collaboration** (`allow_delegation=True` on every agent), and they load
two knowledge sources: `docs/Yelp Data Translation.md` (schema) and
`docs/Exploratory Data Analysis.md` (EDA playbook).

New agents added for the lab (beyond the original three):

- `eda_researcher` — runs EDA probes (distribution / anchors / skew)
- `web_researcher` — optional internet sweep via Serper (only active when
  `SERPER_API_KEY` is set in `.env`)
- `editor` — polishes the final JSON payload
- `crew_manager` — orchestrator for `Process.hierarchical`

If `crewai run 11` fails with `Got unexpected extra argument (11)`, the **upstream** CrewAI CLI is on your PATH instead of this project’s wrapper (both register a `crewai` script). Reinstall this package so its script wins, for example:

```bash
uv pip install -e .
```

You can also use `week5lab-cli run …` (identical entry point) or `uv run run_crew 11` (no `run` subcommand).

Same run via `uv`:

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
- `merge_outputs.json` stores the full run records you want to track, merging each prediction with its ground truth (e.g., `user_id`, `item_id`, `ground_truth.stars`, `ground_truth.review`, and your `predicted` output).

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
