# Week5Lab Crew

Simple CrewAI project for predicting Yelp-style ratings and review text.

## Requirements

- Python `>=3.10,<3.14`
- `uv` installed (`pip install uv`)

## Setup

From the project root:

```bash
uv sync
uv pip install "litellm[proxy]"
```

Create `.env` from `.env.example`, then set your provider keys.

Default setup uses NVIDIA NIM:

```bash
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL_NAME=minimaxai/minimax-m2.7
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
```

## Run

Most common command:

```bash
crewai run 11
```

- `11` means row 11 (1-based) from `data/test_review_subset.json`
- You can also pass direct IDs:

```bash
crewai run '{"user_id":"<USER_ID>","item_id":"<ITEM_ID>"}'
crewai run --user_id "<USER_ID>" --item_id "<ITEM_ID>"
```

### Choose crew mode

Available modes:

| Mode           | Process                | Pattern                                          |
|----------------|------------------------|--------------------------------------------------|
| `sequential`   | `Process.sequential`   | Pattern 1 - Research -> Write -> Edit (default) |
| `collab`       | `Process.sequential`   | Pattern 2 - Collaborative single task           |
| `hierarchical` | `Process.hierarchical` | Pattern 3 - Manager decomposes and delegates    |

Example:

```bash
crewai run --crew collab 3
crewai run --crew hierarchical 3
```

### Run with Flow

```bash
uv run run_flow --crew sequential 3
uv run run_flow --crew collab 3
uv run run_flow --crew hierarchical 3
```

## Output files

- `prediction_output.json`: latest prediction only
- `merge_outputs.json`: history of prediction runs

## Project files to edit

- `src/week5lab1/config/agents.yaml`: agent definitions
- `src/week5lab1/config/tasks.yaml`: task definitions
- `src/week5lab1/crew.py`: crew logic
- `src/week5lab1/main.py`: CLI/input handling
- `src/week5lab1/flow.py`: flow orchestration

## Checklist: What Was Implemented

- [x] Index reuse mechanism to avoid unnecessary re-indexing
- [x] `collab` crew mode (`Process.sequential`)
- [x] `hierarchical` crew mode (`Process.hierarchical`)
- [x] Additional specialized agents (`eda_researcher`, `web_researcher`, `editor`, `crew_manager`)
- [x] CrewAI Flow integration (`run_flow` + `src/week5lab1/flow.py`)
- [x] EDA knowledge source integration for better grounding

## TODO

- [ ] Upgrade to a stronger model and compare hallucination rate
- [ ] Route hierarchical default path through stricter research-then-report chain
- [ ] Add automated evaluation metrics (MAE + review similarity)
- [ ] Add regression tests for identity-lock behavior
- [ ] Add retry/guardrail policy for grounding violations
- [ ] Tune prompts per model family
