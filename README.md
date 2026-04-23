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

- Modify `src/week5lab1/config/agents.yaml` to define your agents
- Modify `src/week5lab1/config/tasks.yaml` to define your tasks
- Modify `src/week5lab1/crew.py` to add your own logic, tools and specific args
- Modify `src/week5lab1/main.py` to add custom inputs for your agents and tasks
- Modify `src/week5lab1/flow.py` to customize CrewAI Flow orchestration

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

### Running via CrewAI Flow (bonus integration)

The project now includes a Flow entrypoint that wraps the same crews and
persists outputs in the same files (`prediction_output.json` and
`merge_outputs.json`).

```bash
uv run run_flow --crew sequential 3
uv run run_flow --crew collab 3
uv run run_flow --crew hierarchical 3
uv run run_flow '{"user_id":"<USER_ID>","item_id":"<ITEM_ID>"}'
```

Flow behavior:

- Parses `--crew` and row index / JSON payload input
- Runs the selected crew mode (`sequential`, `collab`, or `hierarchical`)
- Extracts/normalizes final JSON prediction (`stars`, `review`)
- Writes `prediction_output.json` and appends to `merge_outputs.json`

You can also set the env var `WEEK5LAB1_CREW_MODE=collab` to change the
default without passing `--crew`.

All three crews enable **memory** (`memory=True` on the Crew) and
**collaboration** (`allow_delegation=True` on every agent), and they load
two knowledge sources: `docs/Yelp Data Translation.md` (schema) and
`docs/Exploratory Data Analysis.md` (EDA playbook).

Task config note:

- `tasks.yaml` now keeps both task families:
  - `analyze_user_task` / `analyze_item_task` / `predict_review_task` for the
    main sequential/collab/hierarchical crew pipelines used in `crew.py`
  - `prediction_research_task` / `prediction_report_task` for strict
    research-then-report decomposition experiments

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

## Project Status (Before vs Now)

### Previous State

- Baseline crew logic existed, but task/agent mappings were not fully aligned
  after prompt refactors.
- Pattern coverage was partially present, but runtime wiring caused startup
  errors in some modes.
- Grounding controls were weaker, which made hallucinated business details more
  likely in final predictions.
- No explicit Flow-oriented status summary in the README.

### New Changes Implemented

This repository now reflects all requested lab deliverables (1-4 + bonus):

1. **Index-Reuse mechanism integration**
   - Implemented via Chroma collection reuse (`create_rag_tool` +
     `chroma_collection_exists`) to avoid unnecessary re-indexing.

2. **Crew with `Process.sequential` (Pattern 2 collaborative setup)**
   - `collab` mode is available and runs with delegation-enabled agents.

3. **Crew with `Process.hierarchical` (manager agent)**
   - `hierarchical` mode is available with explicit `crew_manager`
     orchestration.

4. **New/stronger agents**
   - Added and wired specialist agents including:
     `eda_researcher`, `web_researcher`, `editor`, and `crew_manager`.
   - Also included strict research/report specialists for structured pipelines:
     `prediction_researcher` and `prediction_reporting_analyst`.

**Bonus delivered**

- **New knowledge for agents (EDA):**
  `docs/Exploratory Data Analysis.md` is integrated as a knowledge source and
  now includes explicit methodology-only/anti-hallucination guidance.
- **CrewAI Flow integration:**
  `src/week5lab1/flow.py` and `run_flow` entrypoint are implemented so the
  same crews can run through Flow orchestration and persist outputs.

### Future Work (TODO)

- [ ] Upgrade to a stronger model and compare hallucination rate vs current
      baseline.
- [ ] Route hierarchical production path through the stricter
      research-then-report chain by default.
- [ ] Add automated evaluation metrics (MAE for stars + semantic similarity for
      review text) across `merge_outputs.json`.
- [ ] Add regression tests for identity-lock behavior (no wrong business names).
- [ ] Add retry/guardrail policy when output violates grounding constraints.
- [ ] Tune prompts per model family (NIM MiniMax vs alternative providers).

## Current Limitation (Model Quality)

With the current model configuration, the crew can still produce occasional
hallucinated or weakly grounded review details (for example, drifting to an
incorrect business identity or overly generic positive language).

Given time constraints, the current prompt hardening and grounding checks are
the best available mitigation in this iteration. A stronger model will likely
improve faithfulness and calibration, and is the recommended next step when
time allows.

## Support

For support, questions, or feedback regarding the Week5Lab Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
