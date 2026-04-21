"""Reusable RAG tool helpers for the Week5Lab1 crew.

The module exposes:

* :func:`chroma_collection_exists` — sqlite3-based probe that answers
  "is this collection already persisted?" without instantiating a Chroma client
  (sidesteps CrewAI's singleton init and avoids ``chromadb.sqlite3`` write locks).
* :func:`create_rag_tool` — factory that returns a ready-to-use
  :class:`crewai_tools.JSONSearchTool`. On a cold start it triggers the
  full load → chunk → embed pass; on every subsequent run it reattaches
  to the existing collection in well under a second.
* :class:`MyCustomTool` — minimal template kept for parity with the upstream
  ``Rag_Crew_Profiler`` layout so new tools can be added alongside the RAG
  helpers without touching the crew module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Type

from crewai.tools import BaseTool
from crewai_tools import JSONSearchTool
from crewai_tools.tools.json_search_tool.json_search_tool import (
    FixedJSONSearchToolSchema,
)
from pydantic import BaseModel, Field


__all__ = [
    "chroma_collection_exists",
    "create_rag_tool",
    "MyCustomTool",
]


def _chroma_sqlite_path() -> Path | None:
    """Best-effort lookup of the persistent ``chroma.sqlite3`` file."""
    try:
        from crewai.utilities.paths import db_storage_path
    except Exception:
        return None
    return Path(db_storage_path()) / "chroma.sqlite3"


def chroma_collection_exists(collection_name: str) -> bool:
    """Return True if a ChromaDB collection with ``collection_name`` is already persisted.

    We query ``chroma.sqlite3`` directly because opening a :class:`chromadb.Client`
    here would fight CrewAI's global RAG client singleton and any active tool
    holding the DB open.
    """
    db_file = _chroma_sqlite_path()
    if db_file is None or not db_file.is_file():
        return False

    try:
        conn = sqlite3.connect(str(db_file))
        try:
            cur = conn.execute(
                "SELECT id FROM collections WHERE name = ?", (collection_name,)
            )
            return cur.fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def create_rag_tool(
    *,
    json_path: str | Path,
    collection_name: str,
    config: Mapping[str, Any],
    name: str,
    description: str,
) -> JSONSearchTool:
    """Return a :class:`JSONSearchTool` that skips re-indexing on warm boots.

    Cold start → ``JSONSearchTool(json_path=..., collection_name=...)`` pays the
    full indexing cost (hours on CPU for a large JSON).
    Warm start → ``JSONSearchTool(collection_name=...)`` with ``json_path`` omitted
    attaches to the persisted collection in well under a second.

    In both cases the tool's ``args_schema`` is pinned to
    :class:`FixedJSONSearchToolSchema` so the agent only ever passes
    ``search_query`` (the cold-start path would otherwise expose the
    ``json_path`` field and invite the LLM to retrigger the chunk/embed loop).
    """
    path_str = str(json_path)
    if chroma_collection_exists(collection_name):
        tool = JSONSearchTool(collection_name=collection_name, config=dict(config))
    else:
        tool = JSONSearchTool(
            json_path=path_str,
            collection_name=collection_name,
            config=dict(config),
        )

    tool.args_schema = FixedJSONSearchToolSchema
    tool.name = name
    tool.description = description
    generate = getattr(tool, "_generate_description", None)
    if callable(generate):
        generate()
    return tool


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    """Template tool kept for parity with the upstream `Rag_Crew_Profiler` layout."""

    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will "
        "need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        return "this is an example of a tool output, ignore it and move along."
