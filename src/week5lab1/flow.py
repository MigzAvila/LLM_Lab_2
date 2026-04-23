#!/usr/bin/env python
"""CrewAI Flow wrapper for the Week5Lab1 prediction crews."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from crewai.flow.flow import Flow, listen, start

from week5lab1.crew import build_crew


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEST_REVIEW_PATH = _PROJECT_ROOT / "data" / "test_review_subset.json"
_PREDICTION_OUTPUT_PATH = _PROJECT_ROOT / "prediction_output.json"
_MERGE_OUTPUTS_PATH = _PROJECT_ROOT / "merge_outputs.json"
_VALID_CREW_MODES = {"sequential", "collab", "hierarchical"}


def _load_test_rows() -> list[dict]:
    if not _TEST_REVIEW_PATH.is_file():
        raise FileNotFoundError(f"Test file not found: {_TEST_REVIEW_PATH}")
    raw = _TEST_REVIEW_PATH.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No JSONL rows in {_TEST_REVIEW_PATH}")
    return rows


def _resolve_row(cli_arg: str | None) -> dict:
    rows = _load_test_rows()
    if cli_arg is None or cli_arg == "":
        return rows[0]
    if cli_arg.isdigit():
        idx = int(cli_arg)
        if not 1 <= idx <= len(rows):
            raise ValueError(f"Test row index must be 1..{len(rows)}; got {idx}.")
        return rows[idx - 1]

    payload = json.loads(cli_arg)
    if not isinstance(payload, dict) or not payload.get("user_id") or not payload.get("item_id"):
        raise ValueError('JSON payload must be an object with "user_id" and "item_id".')
    return payload


def _parse_cli_args(argv: Iterable[str]) -> tuple[str, str | None]:
    tokens = [t for t in argv if t != "run"]
    crew_mode: str | None = None
    positional: str | None = None

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("--crew", "--mode"):
            if i + 1 >= len(tokens):
                raise ValueError("--crew flag requires a value.")
            crew_mode = tokens[i + 1].strip().lower()
            i += 2
            continue
        if token.startswith("--crew=") or token.startswith("--mode="):
            crew_mode = token.split("=", 1)[1].strip().lower()
            i += 1
            continue
        if positional is None:
            positional = token
        i += 1

    if crew_mode is None:
        crew_mode = "sequential"
    if crew_mode not in _VALID_CREW_MODES:
        valid = ", ".join(sorted(_VALID_CREW_MODES))
        raise ValueError(f"Unknown --crew mode '{crew_mode}'. Valid modes: {valid}.")
    return crew_mode, positional


def _extract_json_from_output(raw_output: Any) -> dict:
    text = str(raw_output).strip().replace("{{", "{").replace("}}", "}")
    match = re.search(r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"stars": None, "review": text, "_parse_error": True}


def _load_merge_outputs() -> list:
    if not _MERGE_OUTPUTS_PATH.is_file():
        return []
    try:
        data = json.loads(_MERGE_OUTPUTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _append_merge_output(row: dict, predicted: dict, *, crew_mode: str) -> None:
    ground_truth = {
        "stars": row.get("stars"),
        "review": row.get("text"),
        "review_id": row.get("review_id"),
        "date": row.get("date"),
    }
    record = {
        "user_id": row.get("user_id"),
        "item_id": row.get("item_id"),
        "crew_mode": crew_mode,
        "ground_truth": ground_truth,
        "predicted": predicted,
    }
    merged = _load_merge_outputs()
    merged.append(record)
    _MERGE_OUTPUTS_PATH.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )


class Week5LabFlowState(BaseModel):
    crew_mode: str = "sequential"
    row: dict = {}
    inputs: dict = {}
    predicted: dict = {}


class Week5LabPredictionFlow(Flow[Week5LabFlowState]):
    """CrewAI Flow that orchestrates a single Week5Lab1 prediction run."""

    @start()
    def load_request(self) -> dict:
        crew_mode, cli_arg = _parse_cli_args(sys.argv[1:])
        row = _resolve_row(cli_arg)
        self.state.crew_mode = crew_mode
        self.state.row = row
        self.state.inputs = {"user_id": row["user_id"], "item_id": row["item_id"]}
        return self.state.inputs

    @listen(load_request)
    def execute_crew(self, _: dict) -> dict:
        crew = build_crew(self.state.crew_mode)
        result = crew.kickoff(inputs=self.state.inputs)
        self.state.predicted = _extract_json_from_output(getattr(result, "raw", result))
        return self.state.predicted

    @listen(execute_crew)
    def persist_outputs(self, predicted: dict) -> dict:
        _PREDICTION_OUTPUT_PATH.write_text(
            json.dumps(predicted, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _append_merge_output(self.state.row, predicted, crew_mode=self.state.crew_mode)
        return predicted


def run_flow() -> dict:
    """Run Week5Lab1 through CrewAI Flow orchestration."""
    flow = Week5LabPredictionFlow()
    return flow.kickoff()
