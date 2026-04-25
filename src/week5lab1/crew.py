"""Yelp recommendation crews.

This module exposes three Crew variants built on the same pool of agents /
tools, so the Week5Lab1 lab can demonstrate all three collaboration
patterns from the lecture:

* :class:`Week5Lab1` — the default **sequential Research -> Write -> Edit**
  pattern (Pattern 1). Context chaining happens through
  ``Process.sequential`` + implicit task dependencies.
* :class:`Week5Lab1CollabCrew` — **Pattern 2: Collaborative Single Task**
  (sequential process, one task, ``allow_delegation=True`` on every agent
  so peers can ask each other questions).
* :class:`Week5Lab1HierarchicalCrew` — **Pattern 3: Hierarchical** with a
  dedicated ``crew_manager`` agent that decomposes a top-level task and
  delegates to specialists via ``Process.hierarchical``.

All three crews share:

* LLM: NVIDIA NIM (OpenAI-compatible endpoint at
  ``https://integrate.api.nvidia.com/v1``) by default, driven by the
  ``NVIDIA_*`` env vars. Override with ``LLM_PROVIDER`` / ``MODEL`` to go
  back to Ollama or any other LiteLLM provider.
* Embeddings: ``BAAI/bge-small-en-v1.5`` via sentence-transformers (CPU/GPU).
* Vector store: CrewAI's bundled ChromaDB — reused across runs through the
  smart sqlite3 cache probe in :mod:`week5lab1.tools.custom_tool`.
* Memory: ``memory=True`` on the Crew so agents retain context across tasks.
* Caching: default CrewAI tool result caching (``cache=True`` per agent).
* Knowledge: the Yelp schema doc **and** the EDA playbook are injected as
  ``StringKnowledgeSource`` objects.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# The BAAI/bge-small-en-v1.5 weights are cached under ~/.cache/huggingface.
# Force offline mode so sentence-transformers / huggingface_hub don't retry
# HEAD requests against huggingface.co when DNS is unreachable (common in WSL).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Some crewai-tools versions validate ``OPENAI_API_KEY`` at import time even when
# the tool never calls OpenAI; a placeholder keeps the import path clean.
os.environ.setdefault("OPENAI_API_KEY", "NA")
_placeholder_key = os.environ.get("OPENAI_API_KEY", "").strip().upper()
if _placeholder_key in ("", "NONE"):
    os.environ["OPENAI_API_KEY"] = "NA"


from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.agents.agent_builder.base_agent import BaseAgent  # noqa: E402
from crewai.knowledge.source.string_knowledge_source import (  # noqa: E402
    StringKnowledgeSource,
)
from crewai.project import CrewBase, agent, crew, task  # noqa: E402

from week5lab1.tools.custom_tool import create_rag_tool  # noqa: E402


# ---------------------------------------------------------------------------
# LLM selection
#
# Default provider is local Ollama, driven by ``.env``:
#
#     LLM_PROVIDER=ollama
#     MODEL=ollama/gemma4:26b
#     OLLAMA_API_BASE=http://127.0.0.1:11434
#
# To use NVIDIA NIM instead, set ``LLM_PROVIDER=nvidia`` and fill the
# ``NVIDIA_*`` variables in ``.env``.
# ---------------------------------------------------------------------------

_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def _build_llm() -> LLM:
    """Construct the shared :class:`crewai.LLM` used by every agent."""
    if _LLM_PROVIDER == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=nvidia but NVIDIA_API_KEY is not set in .env"
            )
        model_name = os.getenv("NVIDIA_MODEL_NAME", "minimaxai/minimax-m2.7").strip()
        base_url = os.getenv(
            "NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"
        ).strip()

        # NVIDIA's API Catalog endpoint is OpenAI-compatible, so we route via
        # LiteLLM's ``openai/`` provider with an explicit base URL. We also
        # mirror the key into ``OPENAI_API_KEY`` because some LiteLLM code
        # paths read it from the process env even when ``api_key`` is passed.
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ.setdefault("OPENAI_API_BASE", base_url)

        litellm_model = (
            model_name if model_name.startswith("openai/") else f"openai/{model_name}"
        )
        os.environ["MODEL"] = litellm_model

        return LLM(
            model=litellm_model,
            api_key=api_key,
            base_url=base_url,
        )

    # Generic fallback: honour whatever ``MODEL`` says (Ollama, Groq, etc.).
    model_name = os.getenv("MODEL") or "ollama/gemma4:26b"
    os.environ.setdefault("MODEL", model_name)
    os.environ.setdefault("OLLAMA_API_BASE", os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434"))
    return LLM(model=model_name)


CREW_LLM: LLM = _build_llm()


_DATA_DIR = _PROJECT_ROOT / "data"
_DOCS_DIR = _PROJECT_ROOT / "docs"
_EMBEDDING_DEVICE = os.getenv("CREW_EMBEDDING_DEVICE", "cpu")

RAG_CONFIG: dict = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "device": _EMBEDDING_DEVICE,
        },
    }
}

CREW_EMBEDDER: dict = RAG_CONFIG["embedding_model"]


# ---------------------------------------------------------------------------
# Shared RAG tools (built once at import time; ChromaDB is reused across runs)
# ---------------------------------------------------------------------------

# Collection names mirror the upstream "Index Collection Reference" table in
# docs/RAG_Index_Student_Guide_EN.md so a pre-built chroma.sqlite3 archive
# produced against Rag_Crew_Profiler drops in as a warm cache here.
# Source-data file names differ (the lab ships ``*_subset.json`` instead of
# ``filtered_*.json`` / ``test_review.json``) but the content role is the same.
USER_COLLECTION = "benchmark_true_fresh_index_Filtered_User_1"
ITEM_COLLECTION = "benchmark_true_fresh_index_Filtered_Item_1"
REVIEW_COLLECTION = "benchmark_true_fresh_index_Filtered_Review_1"


user_rag_tool = create_rag_tool(
    json_path=_DATA_DIR / "user_subset.json",
    collection_name=USER_COLLECTION,
    config=RAG_CONFIG,
    name="search_user_profile_data",
    description=(
        "Searches the user profile database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the review habits and average stars for user _BcWyKQL16?'. "
        "Do NOT pass raw user_id or JSON objects directly."
    ),
)

item_rag_tool = create_rag_tool(
    json_path=_DATA_DIR / "item_subset.json",
    collection_name=ITEM_COLLECTION,
    config=RAG_CONFIG,
    name="search_restaurant_feature_data",
    description=(
        "Searches the restaurant/business database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the categories, location, and star rating for business abc123?'. "
        "Do NOT pass raw item_id or JSON objects directly."
    ),
)

review_rag_tool = create_rag_tool(
    json_path=_DATA_DIR / "review_subset.json",
    collection_name=REVIEW_COLLECTION,
    config=RAG_CONFIG,
    name="search_historical_reviews_data",
    description=(
        "Searches historical review texts using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'Find past reviews written by user _BcWyKQL16 about food quality and service'. "
        "Do NOT pass raw user_id, item_id, or JSON objects directly."
    ),
)


def _build_web_search_tool():
    """Return a Serper web-search tool if ``SERPER_API_KEY`` is configured, else None.

    The web_researcher agent is only useful when an external search provider
    is wired up. When no key is present we return None and the agent simply
    runs toolless (it will politely decline delegations in that case, per
    its backstory).
    """
    if not os.getenv("SERPER_API_KEY"):
        return None
    try:
        from crewai_tools import SerperDevTool  # type: ignore
    except Exception:
        return None
    try:
        return SerperDevTool()
    except Exception:
        return None


_WEB_SEARCH_TOOL = _build_web_search_tool()


# ---------------------------------------------------------------------------
# Knowledge sources
# ---------------------------------------------------------------------------


def _load_string_knowledge(path: Path, source_label: str) -> StringKnowledgeSource:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        content = f"{source_label} not available."
    return StringKnowledgeSource(
        content=content,
        metadata={"source": source_label},
    )


def _load_schema_knowledge() -> StringKnowledgeSource:
    """Inject the Yelp schema crib sheet so agents interpret retrieved fields correctly."""
    return _load_string_knowledge(
        _DOCS_DIR / "Yelp Data Translation.md", "Yelp Schema Definition"
    )


def _load_eda_knowledge() -> StringKnowledgeSource:
    """Inject the EDA playbook so agents share a common calibration vocabulary."""
    return _load_string_knowledge(
        _DOCS_DIR / "Exploratory Data Analysis.md",
        "Exploratory Data Analysis Playbook",
    )


def _crew_knowledge_sources() -> List[StringKnowledgeSource]:
    return [_load_schema_knowledge(), _load_eda_knowledge()]


# ===========================================================================
# Crew 1 — Pattern 1: Sequential Research -> Write -> Edit (the default)
# ===========================================================================


@CrewBase
class Week5Lab1:
    """Yelp Recommendation Crew: User Profiler -> Restaurant Analyst -> Prediction Expert."""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["user_analyst"],  # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["item_analyst"],  # type: ignore[index]
            tools=[item_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_researcher"],  # type: ignore[index]
            tools=[user_rag_tool, item_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_reporting_analyst"],  # type: ignore[index]
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_modeler"],  # type: ignore[index]
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def calibrator(self) -> Agent:
        return Agent(
            config=self.agents_config["calibrator"],  # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_user_task"])  # type: ignore[index]

    @task
    def analyze_item_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_item_task"])  # type: ignore[index]

    @task
    def calibrate_user_task(self) -> Task:
        return Task(config=self.tasks_config["calibrate_user_task"])  # type: ignore[index]

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["predict_review_task"],  # type: ignore[index]
            output_file="prediction_output.json",
        )

    @task
    def prediction_research_task(self) -> Task:
        return Task(config=self.tasks_config["prediction_research_task"])  # type: ignore[index]

    @task
    def prediction_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["prediction_report_task"],  # type: ignore[index]
            output_file="prediction_output.json",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=_crew_knowledge_sources(),
            embedder=CREW_EMBEDDER,
            memory=True,
            cache=True,
            verbose=True,
        )


# ===========================================================================
# Crew 2 — Pattern 2: Collaborative Single Task (sequential process)
# ===========================================================================


@CrewBase
class Week5Lab1CollabCrew:
    """Pattern 2: collaborative specialists under ``Process.sequential``.

    Mirrors the sequential crew's three-task lineup
    (``analyze_user_task`` -> ``analyze_item_task`` -> ``predict_review_task``)
    so the final output is emitted by ``prediction_modeler`` — same clean
    JSON schema as the sequential and hierarchical crews. The
    "collaborative" flavour of Pattern 2 is still there because every agent
    (including the extra ``eda_researcher`` / ``web_researcher``) has
    ``allow_delegation=True`` in :file:`config/agents.yaml`, so an analyst
    can still ask a peer a focused question mid-task via CrewAI's built-in
    delegation tools.

    Why not a single-task setup? When the single-task owner has no direct
    tools, it is forced onto the delegation meta-tools ("Delegate work to
    coworker" / "Ask question to coworker"). The NVIDIA MiniMax NIM emits
    those as its native ``<minimax:tool_call>`` XML, which LiteLLM does
    not translate — the XML leaks into the final output and the
    prediction is lost. Pinning each step to a specialist with its own
    RAG tools routes tool calls through the OpenAI-style function-call
    path, which MiniMax handles correctly.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["user_analyst"],  # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["item_analyst"],  # type: ignore[index]
            tools=[item_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_researcher"],  # type: ignore[index]
            tools=[user_rag_tool, item_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_reporting_analyst"],  # type: ignore[index]
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def eda_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["eda_researcher"],  # type: ignore[index]
            tools=[user_rag_tool, item_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def web_researcher(self) -> Agent:
        tools = [_WEB_SEARCH_TOOL] if _WEB_SEARCH_TOOL is not None else []
        return Agent(
            config=self.agents_config["web_researcher"],  # type: ignore[index]
            tools=tools,
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_modeler"],  # type: ignore[index]
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def calibrator(self) -> Agent:
        return Agent(
            config=self.agents_config["calibrator"],  # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    # Reuse the sequential task lineup so the final output is emitted by
    # ``prediction_modeler`` via the reliable OpenAI-style tool-call path.
    # Peer collaboration still happens mid-task via ``allow_delegation=True``
    # on the agents in :file:`config/agents.yaml`.

    @task
    def analyze_user_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_user_task"])  # type: ignore[index]

    @task
    def analyze_item_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_item_task"])  # type: ignore[index]

    @task
    def calibrate_user_task(self) -> Task:
        return Task(config=self.tasks_config["calibrate_user_task"])  # type: ignore[index]

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["predict_review_task"],  # type: ignore[index]
            output_file="prediction_output.json",
        )

    @task
    def prediction_research_task(self) -> Task:
        return Task(config=self.tasks_config["prediction_research_task"])  # type: ignore[index]

    @task
    def prediction_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["prediction_report_task"],  # type: ignore[index]
            output_file="prediction_output.json",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=_crew_knowledge_sources(),
            embedder=CREW_EMBEDDER,
            memory=True,
            cache=True,
            verbose=True,
        )


# ===========================================================================
# Crew 3 — Pattern 3: Hierarchical (manager agent decomposes & delegates)
# ===========================================================================


@CrewBase
class Week5Lab1HierarchicalCrew:
    """Pattern 3: ``crew_manager`` decomposes a top-level task and delegates."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # NOTE: the manager agent is passed via ``manager_agent=...`` and must NOT
    # appear in the ``agents`` list; therefore it is built inside ``crew()``.

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["user_analyst"],  # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["item_analyst"],  # type: ignore[index]
            tools=[item_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_researcher"],  # type: ignore[index]
            tools=[user_rag_tool, item_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_reporting_analyst"],  # type: ignore[index]
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def eda_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["eda_researcher"],  # type: ignore[index]
            tools=[user_rag_tool, item_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def web_researcher(self) -> Agent:
        tools = [_WEB_SEARCH_TOOL] if _WEB_SEARCH_TOOL is not None else []
        return Agent(
            config=self.agents_config["web_researcher"],  # type: ignore[index]
            tools=tools,
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_modeler"],  # type: ignore[index]
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def calibrator(self) -> Agent:
        return Agent(
            config=self.agents_config["calibrator"],  # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=CREW_LLM,
            verbose=True,
        )

    @agent
    def editor(self) -> Agent:
        return Agent(
            config=self.agents_config["editor"],  # type: ignore[index]
            llm=CREW_LLM,
            verbose=True,
        )

    # Mirror the sequential crew's task lineup so the final output is emitted
    # by ``prediction_modeler`` (not the manager's narration). Under
    # ``Process.hierarchical`` the manager still orchestrates and can
    # side-delegate to ``eda_researcher`` / ``editor`` / ``web_researcher``
    # via the built-in "Delegate work to coworker" / "Ask question to
    # coworker" tools, but the concrete per-step output is owned by the
    # agent pinned in ``tasks.yaml``.

    @task
    def analyze_user_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_user_task"])  # type: ignore[index]

    @task
    def analyze_item_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_item_task"])  # type: ignore[index]

    @task
    def calibrate_user_task(self) -> Task:
        return Task(config=self.tasks_config["calibrate_user_task"])  # type: ignore[index]

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["predict_review_task"],  # type: ignore[index]
            output_file="prediction_output.json",
        )

    @task
    def prediction_research_task(self) -> Task:
        return Task(config=self.tasks_config["prediction_research_task"])  # type: ignore[index]

    @task
    def prediction_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["prediction_report_task"],  # type: ignore[index]
            output_file="prediction_output.json",
        )

    def _build_manager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["crew_manager"],  # type: ignore[index]
            llm=CREW_LLM,
            verbose=True,
            allow_delegation=True,
        )

    @crew
    def crew(self) -> Crew:
        manager = self._build_manager_agent()
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=manager,
            knowledge_sources=_crew_knowledge_sources(),
            embedder=CREW_EMBEDDER,
            memory=True,
            cache=True,
            verbose=True,
        )


# ---------------------------------------------------------------------------
# Convenience factory for main.py's CLI selector.
# ---------------------------------------------------------------------------

_CREW_REGISTRY = {
    "sequential": Week5Lab1,
    "collab": Week5Lab1CollabCrew,
    "hierarchical": Week5Lab1HierarchicalCrew,
}


def build_crew(mode: Optional[str] = None) -> Crew:
    """Return a ready-to-kick-off Crew for the requested mode.

    ``mode`` accepts ``"sequential"`` (default, Pattern 1),
    ``"collab"`` (Pattern 2), or ``"hierarchical"`` (Pattern 3). It also
    honours the ``WEEK5LAB1_CREW_MODE`` env var when ``mode`` is None.
    """
    key = (mode or os.getenv("WEEK5LAB1_CREW_MODE") or "sequential").strip().lower()
    try:
        cls = _CREW_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_CREW_REGISTRY))
        raise ValueError(
            f"Unknown crew mode '{key}'. Valid modes: {valid}."
        ) from exc
    return cls().crew()
