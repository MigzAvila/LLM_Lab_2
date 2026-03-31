#!/usr/bin/env python
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

from week5lab.crew import Week5Lab

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TEST_REVIEW_PATH = _PROJECT_ROOT / "data" / "test_review_subset.json"
_PREDICTION_OUTPUTS_PATH = _PROJECT_ROOT / "prediction_outputs.json"
_DEFAULT_PAIR = {
    "user_id": "2YKkLFeOx-0zRcWp0KUv_Q",
    "item_id": "vJdsF2pRH6pZZ16snLHSaw",
}


def _load_test_subset_rows() -> list[dict]:
    if not _TEST_REVIEW_PATH.is_file():
        raise FileNotFoundError(
            f"Test file not found: {_TEST_REVIEW_PATH}. "
            "Add data/test_review_subset.json or pass explicit JSON."
        )
    raw = _TEST_REVIEW_PATH.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No JSONL rows in {_TEST_REVIEW_PATH}")
    return rows


def _inputs_from_payload(payload: dict) -> dict:
    uid = payload.get("user_id")
    iid = payload.get("item_id")
    if not uid or not iid:
        raise ValueError('Payload must include non-empty "user_id" and "item_id".')
    return {
        "user_id": uid,
        "item_id": iid,
        "current_year": str(datetime.now().year),
    }


def _read_prediction_json_file() -> dict | None:
    path = _PROJECT_ROOT / "prediction_output.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "stars" in data and "review" in data:
        return {"stars": data["stars"], "review": data["review"]}
    return None


def _load_prediction_outputs_list() -> list:
    if not _PREDICTION_OUTPUTS_PATH.is_file():
        return []
    try:
        data = json.loads(_PREDICTION_OUTPUTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _append_prediction_solution(
    *,
    user_id: str,
    item_id: str,
    predicted: dict | None,
    actual_stars=None,
) -> None:
    """Append one run to prediction_outputs.json (CrewAI overwrites prediction_output.json each time)."""
    record = {
        "user_id": user_id,
        "item_id": item_id,
        "actual_stars": actual_stars,
        "predicted": predicted,
    }
    data = _load_prediction_outputs_list()
    data.append(record)
    _PREDICTION_OUTPUTS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _run_all_test_pairs() -> None:
    """Run the crew for every (user_id, item_id) in test_review_subset.json (JSONL)."""
    rows = _load_test_subset_rows()

    print(
        f"Running {len(rows)} (user_id, item_id) pair(s) from {_TEST_REVIEW_PATH.name} …"
    )
    _PREDICTION_OUTPUTS_PATH.write_text("[]", encoding="utf-8")
    crew_factory = Week5Lab()
    summaries = []
    for row in rows:
        inputs = _inputs_from_payload(
            {"user_id": row["user_id"], "item_id": row["item_id"]}
        )
        crew_factory.crew().kickoff(inputs=inputs)
        pred = _read_prediction_json_file()
        _append_prediction_solution(
            user_id=row["user_id"],
            item_id=row["item_id"],
            predicted=pred,
            actual_stars=row.get("stars"),
        )
        summaries.append(
            {
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "actual_stars": row.get("stars"),
                "predicted": pred,
            }
        )

    out_path = _PROJECT_ROOT / "test_eval_summary.json"
    out_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path}")
    print(f"Appended all runs to {_PREDICTION_OUTPUTS_PATH.name}")


def _run_single_prediction(payload: dict):
    """One kickoff for user_id/item_id in ``payload``; append to prediction_outputs when valid."""
    inputs = _inputs_from_payload(payload)

    try:
        result = Week5Lab().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}") from e

    out = _read_prediction_json_file()
    if out is not None:
        _append_prediction_solution(
            user_id=inputs["user_id"],
            item_id=inputs["item_id"],
            predicted=out,
            actual_stars=payload.get("stars"),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"Appended run to {_PREDICTION_OUTPUTS_PATH.name}")
        return result

    raw = getattr(result, "raw", None) or str(result)
    print(raw)
    return result


def run():
    """
    Run the review-prediction crew for one (user_id, item_id) pair.

    Usage:
      crewai run N | uv run run_crew N
        → row N from data/test_review_subset.json (1-based JSONL line index)

      crewai run '{"user_id":"...","item_id":"..."}' | uv run run_crew '...'
        → explicit pair; prints prediction_output.json contents when valid

      uv run test_eval
        → full subset (all rows); long-running; writes test_eval_summary.json
    """
    cli_arg = sys.argv[1].strip() if len(sys.argv) >= 2 else ""
    
    if not cli_arg:
        raise Exception(
            "Missing argument: pass a 1-based test row id (order in "
            "data/test_review_subset.json) or a JSON object with user_id and item_id. "
            "Example: crewai run 3. For the full subset use: uv run test_eval"
        )

    if cli_arg.isdigit():
        try:
            rows = _load_test_subset_rows()
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}") from e
        idx = int(cli_arg)
        if idx < 1 or idx > len(rows):
            raise Exception(
                f"Test row index must be from 1 to {len(rows)} "
                f"(1-based order in {_TEST_REVIEW_PATH.name}); got {idx}."
            )
        row = rows[idx - 1]
        print(
            f"Running test subset row {idx}/{len(rows)} "
            f"(user_id={row['user_id']}, item_id={row['item_id']}) …"
        )
        payload = {
            "user_id": row["user_id"],
            "item_id": row["item_id"],
            "stars": row.get("stars"),
        }
        return _run_single_prediction(payload)

    try:
        payload = json.loads(cli_arg)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON argument: {e}") from e

    if not isinstance(payload, dict):
        raise Exception(
            "CLI argument must be a 1-based row number or a JSON object with user_id and item_id."
        )

    return _run_single_prediction(payload)


def run_test_eval():
    """
    Same as ``run()`` with no CLI args: evaluate every row in test_review_subset.json.

    The crew and RAG tools never load that file; it is only used to pick pairs and
    record actual_stars vs predictions in test_eval_summary.json.
    """
    _run_all_test_pairs()


def train():
    inputs = {
        "user_id": _DEFAULT_PAIR["user_id"],
        "item_id": _DEFAULT_PAIR["item_id"],
        "current_year": str(datetime.now().year),
    }
    try:
        Week5Lab().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}") from e


def replay():
    try:
        Week5Lab().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}") from e


def test():
    inputs = {
        "user_id": _DEFAULT_PAIR["user_id"],
        "item_id": _DEFAULT_PAIR["item_id"],
        "current_year": str(datetime.now().year),
    }
    try:
        Week5Lab().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}") from e


def run_with_trigger():
    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Please provide JSON payload as argument."
        )
    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "user_id": trigger_payload.get("user_id", ""),
        "item_id": trigger_payload.get("item_id", ""),
        "current_year": str(datetime.now().year),
    }
    try:
        return Week5Lab().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}") from e


def crewai_entry() -> None:
    av = sys.argv
    # Intercept BEFORE Click
    if len(av) >= 2 and av[1] == "run":
        # Remove "run" so Click doesn't process it
        args = av[2:]

        if not args:
            print(
                "error: crewai run requires input:\n"
                "  crewai run 3\n"
                "  crewai run '{\"user_id\":\"123\",\"item_id\":\"456\"}'\n"
                "  crewai run --user_id 123 --item_id 456",
                file=sys.stderr,
            )
            raise SystemExit(2)

        arg = args[0]

        # JSON
        if arg.lstrip().startswith("{"):
            sys.argv = ["run_crew", arg]
            run()
            return

        # Row ID
        if arg.isdigit():
            sys.argv = ["run_crew", arg]
            run()
            return

        # Named params
        if arg.startswith("--"):
            parsed = {}
            key = None

            for token in args:
                if token.startswith("--"):
                    key = token.lstrip("-")
                    parsed[key] = None
                else:
                    if key is None:
                        raise ValueError("Invalid argument format")
                    parsed[key] = token
                    key = None

            json_arg = json.dumps(parsed)
            sys.argv = ["run_crew", json_arg]
            run()
            return

    # Fallback to CrewAI CLI
    from crewai.cli.cli import crewai as crewai_click_group
    crewai_click_group()
