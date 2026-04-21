#!/usr/bin/env python
"""Entry point for the Week5Lab1 Yelp review-prediction crew.

Usage::

    uv run run_crew                          # sequential crew, first test row
    uv run run_crew 3                        # sequential crew, row 3
    uv run run_crew --crew collab 3          # Pattern 2 collaborative crew
    uv run run_crew --crew hierarchical 3    # Pattern 3 hierarchical crew
    uv run run_crew '{"user_id":"...","item_id":"..."}'

The crew writes its final JSON to ``prediction_output.json`` at the project
root via the ``output_file`` on the terminating task; we additionally parse
the raw LLM output, rewrite ``prediction_output.json`` with a validated JSON
object, and append a record (prediction + ground truth) to
``merge_outputs.json`` so downstream evaluation scripts can compare against
the test subset.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

from week5lab1.crew import build_crew


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEST_REVIEW_PATH = _PROJECT_ROOT / "data" / "test_review_subset.json"
_PREDICTION_OUTPUT_PATH = _PROJECT_ROOT / "prediction_output.json"
_MERGE_OUTPUTS_PATH = _PROJECT_ROOT / "merge_outputs.json"

_VALID_CREW_MODES = {"sequential", "collab", "hierarchical"}


def _load_test_rows() -> list[dict]:
    """Read ``data/test_review_subset.json`` as a JSONL list of dicts."""
    if not _TEST_REVIEW_PATH.is_file():
        raise FileNotFoundError(f"Test file not found: {_TEST_REVIEW_PATH}")
    raw = _TEST_REVIEW_PATH.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No JSONL rows in {_TEST_REVIEW_PATH}")
    return rows


def _resolve_row(cli_arg: str | None) -> dict:
    """Translate a CLI token into a full test row (or synthetic row from JSON)."""
    rows = _load_test_rows()

    if cli_arg is None or cli_arg == "":
        return rows[0]
    if cli_arg.isdigit():
        idx = int(cli_arg)
        if not 1 <= idx <= len(rows):
            raise ValueError(
                f"Test row index must be 1..{len(rows)}; got {idx}."
            )
        return rows[idx - 1]

    try:
        payload = json.loads(cli_arg)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON argument: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("user_id") or not payload.get("item_id"):
        raise ValueError(
            'JSON payload must be an object with "user_id" and "item_id".'
        )
    return payload


def _extract_json_from_output(raw_output: Any) -> dict:
    """Pull a clean ``{"stars": float, "review": str}`` from the LLM's raw output."""
    text = str(raw_output).strip()
    text = text.replace("{{", "{").replace("}}", "}")

    match = re.search(
        r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL
    )
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
    """Append one ``{ground_truth, predicted}`` record to ``merge_outputs.json``."""
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
    data = _load_merge_outputs()
    data.append(record)
    _MERGE_OUTPUTS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _parse_cli_args(argv: Iterable[str]) -> tuple[str, str | None]:
    """Extract ``(crew_mode, positional_arg)`` from argv.

    Supported forms:
        --crew collab
        --crew=collab
        <positional>     (row index or JSON payload)

    The positional arg is optional. ``crew_mode`` defaults to
    ``WEEK5LAB1_CREW_MODE`` then ``"sequential"``.
    """
    tokens = [t for t in argv if t != "run"]  # tolerate 'crewai run ...'
    crew_mode: str | None = None
    positional: str | None = None

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("--crew", "--mode"):
            if i + 1 >= len(tokens):
                raise ValueError("--crew flag requires a value.")
            crew_mode = tokens[i + 1].strip().lower()
            i += 2
            continue
        if t.startswith("--crew="):
            crew_mode = t.split("=", 1)[1].strip().lower()
            i += 1
            continue
        if t.startswith("--mode="):
            crew_mode = t.split("=", 1)[1].strip().lower()
            i += 1
            continue
        if positional is None:
            positional = t
            i += 1
            continue
        i += 1

    if crew_mode is None:
        crew_mode = os.getenv("WEEK5LAB1_CREW_MODE", "sequential").strip().lower()
    if crew_mode not in _VALID_CREW_MODES:
        valid = ", ".join(sorted(_VALID_CREW_MODES))
        raise ValueError(
            f"Unknown --crew mode '{crew_mode}'. Valid modes: {valid}."
        )
    return crew_mode, positional


def run() -> dict:
    """Run the crew for one ``(user_id, item_id)`` pair and persist outputs."""
    crew_mode, cli_arg = _parse_cli_args(sys.argv[1:])

    row = _resolve_row(cli_arg)
    inputs = {"user_id": row["user_id"], "item_id": row["item_id"]}
    print(
        f"[week5lab1] crew_mode={crew_mode} | "
        f"user={inputs['user_id']} | item={inputs['item_id']}"
    )

    try:
        crew_obj = build_crew(crew_mode)
        result = crew_obj.kickoff(inputs=inputs)
    except Exception as exc:
        raise RuntimeError(f"Crew run failed: {exc}") from exc

    predicted = _extract_json_from_output(getattr(result, "raw", result))
    _PREDICTION_OUTPUT_PATH.write_text(
        json.dumps(predicted, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _append_merge_output(row, predicted, crew_mode=crew_mode)

    print("\n=== Prediction Completed ===")
    print(f"Crew mode: {crew_mode}")
    print(f"Stars: {predicted.get('stars')}")
    review_preview = str(predicted.get("review", ""))[:100]
    print(f"Review: {review_preview}...")
    print(f"Prediction written to {_PREDICTION_OUTPUT_PATH.name}")
    print(f"Appended run to {_MERGE_OUTPUTS_PATH.name}")
    return predicted


def train() -> None:
    """CrewAI training harness placeholder."""
    row = _resolve_row(None)
    inputs = {"user_id": row["user_id"], "item_id": row["item_id"]}
    try:
        build_crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )
    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc


def replay() -> None:
    """Replay a previously recorded crew task."""
    try:
        build_crew().replay(task_id=sys.argv[1])
    except Exception as exc:
        raise RuntimeError(f"Replay failed: {exc}") from exc


def test() -> None:
    """CrewAI evaluation harness placeholder."""
    row = _resolve_row(None)
    inputs = {"user_id": row["user_id"], "item_id": row["item_id"]}
    try:
        build_crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )
    except Exception as exc:
        raise RuntimeError(f"Test run failed: {exc}") from exc


def run_with_trigger() -> Any:
    """Run the crew from a JSON trigger payload (``user_id`` / ``item_id``).

    The payload may optionally carry a ``crew_mode`` key
    (``"sequential"`` | ``"collab"`` | ``"hierarchical"``).
    """
    if len(sys.argv) < 2:
        raise RuntimeError("No trigger payload provided.")
    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON payload provided as argument.") from exc

    inputs = {
        "user_id": payload.get("user_id", ""),
        "item_id": payload.get("item_id", ""),
    }
    crew_mode = str(payload.get("crew_mode", "sequential")).lower()
    if crew_mode not in _VALID_CREW_MODES:
        valid = ", ".join(sorted(_VALID_CREW_MODES))
        raise RuntimeError(
            f"Unknown crew_mode '{crew_mode}' in payload. Valid: {valid}."
        )
    return build_crew(crew_mode).kickoff(inputs=inputs)
