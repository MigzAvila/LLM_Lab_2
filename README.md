# LLM Lab Crew

CrewAI project for predicting Yelp-style star ratings and review text from a `(user_id, item_id)` pair, using RAG over local JSON subsets, optional EDA knowledge, and a dedicated calibration step before the final JSON prediction.

## Requirements

- Python `>=3.10,<3.14`
- `uv` installed (`pip install uv`)
- **Default LLM:** [Ollama](https://ollama.com) with `gemma4:26b` pulled locally (`ollama pull gemma4:26b`)

## Setup

From the project root:

```bash
uv sync
uv pip install "litellm[proxy]"
```

Copy `.env.example` to `.env` and adjust if needed. Defaults target local Ollama:

```bash
LLM_PROVIDER=ollama
MODEL=ollama/gemma4:26b
OLLAMA_API_BASE=http://127.0.0.1:11434
```

Ensure Ollama is running and the model is available:

```bash
ollama serve   # if not already running as a service
ollama pull gemma4:26b
ollama list
```

### Optional: NVIDIA NIM instead of Ollama

Set in `.env`:

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

| Mode           | Process                 | Pattern |
|----------------|-------------------------|---------|
| `sequential`   | `Process.sequential`    | User profile → item profile → **calibration** → final JSON prediction (default) |
| `collab`       | `Process.sequential`    | Same task chain with delegation enabled across agents |
| `hierarchical` | `Process.hierarchical`  | Manager orchestrates the same specialists |

Examples:

```bash
crewai run --crew collab 3
crewai run --crew hierarchical 3
```

### Run with Flow

Same inputs and crew flags; orchestration goes through `src/week5lab1/flow.py`:

```bash
uv run run_flow --crew sequential 3
uv run run_flow --crew collab 3
uv run run_flow --crew hierarchical 3
```

## Pipeline overview

1. **User analyst** — RAG over user and review corpora; profile and numeric anchors.
2. **Item analyst** — RAG over business and reviews; identity-locked business snapshot.
3. **Calibrator** — `calibrate_user_task`: prior and adjustment band so the final step does not collapse to generic mid-range stars (inspired by a dedicated calibration agent pattern).
4. **Prediction modeler** — Single JSON object: `stars` and `review`, grounded in prior tasks.

Optional agents (`eda_researcher`, `web_researcher`, `editor`, `crew_manager`) participate mainly in collaborative and hierarchical modes.

## Output files

- `prediction_output.json`: latest prediction only
- `merge_outputs.json`: history of runs (ground truth from the test row when applicable, plus predicted payload and `crew_mode`)

## Project files to edit

- `src/week5lab1/config/agents.yaml`: agent definitions (including `calibrator`)
- `src/week5lab1/config/tasks.yaml`: task definitions (including `calibrate_user_task`)
- `src/week5lab1/crew.py`: crew factories, LLM selection, RAG tools, knowledge sources
- `src/week5lab1/main.py`: CLI and merge output handling
- `src/week5lab1/flow.py`: Flow orchestration for `run_flow`

## Checklist: What was implemented

- [x] Index reuse (Chroma sqlite probe) to avoid unnecessary re-indexing
- [x] `collab` and `hierarchical` crew modes
- [x] Specialized agents: `eda_researcher`, `web_researcher`, `editor`, `crew_manager`
- [x] **`calibrator` agent** and **`calibrate_user_task`** before final prediction
- [x] CrewAI Flow (`run_flow` + `flow.py`)
- [x] EDA + schema string knowledge sources
- [x] **Default LLM:** Ollama `gemma4:26b` (override via `.env` for NVIDIA or other LiteLLM backends)

## TODO

- [ ] Compare prediction quality: Ollama `gemma4:26b` vs NVIDIA (or other) on the same rows
- [ ] Route hierarchical default path through a stricter research-then-report chain if needed
- [ ] Add automated evaluation metrics (MAE + review similarity) over `merge_outputs.json`
- [ ] Add regression tests for identity-lock and calibration behavior
- [ ] Add retry/guardrail policy for grounding violations
- [ ] Tune prompts per model family (Ollama vs cloud APIs)
