from pathlib import Path
from typing import List, cast

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.rag.chromadb.config import ChromaDBConfig
from crewai.rag.chromadb.types import ChromaEmbeddingFunctionWrapper
from crewai.rag.config.utils import set_rag_config
from crewai.rag.embeddings.factory import build_embedder
from crewai_tools import JSONSearchTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# CrewAI reads MODEL / MODEL_NAME / OPENAI_MODEL_NAME for LiteLLM (e.g. ollama/llama3.1:8b).

# Drop placeholder OpenAI keys (they are sent verbatim and cause 401s on upsert).
_oa = os.environ.get("OPENAI_API_KEY")
if _oa is not None and _oa.strip().upper() in ("NA", "NONE", ""):
    os.environ.pop("OPENAI_API_KEY", None)

# sentence-transformers for RAG / crew knowledge: default cuda (override with CREW_EMBEDDING_DEVICE=cpu if needed).
_EMBEDDING_DEVICE = os.getenv("CREW_EMBEDDING_DEVICE", "cuda")

# Local embeddings for JSON RAG tools (matches prior bge-small setup).
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "device": _EMBEDDING_DEVICE,
        },
    }
}

# Same spec for crew knowledge + any code path that uses the global RAG client.
_CREW_KNOWLEDGE_EMBEDDER = rag_config["embedding_model"]


def _configure_local_rag_embeddings() -> None:
    """Use local sentence-transformers for Chroma everywhere (no OpenAI API for embeddings)."""
    ef = build_embedder(_CREW_KNOWLEDGE_EMBEDDER)  # type: ignore[arg-type]
    set_rag_config(
        ChromaDBConfig(
            embedding_function=cast(ChromaEmbeddingFunctionWrapper, ef),
        )
    )


def _drop_legacy_crew_knowledge_chroma_collection() -> None:
    """Remove ``knowledge_crew`` if it was built with another embedder (e.g. OpenAI).

    Avoid ``ChromaDBConfig()`` here: its default factory instantiates OpenAIEmbeddingFunction
    and requires OPENAI_API_KEY.
    """
    try:
        import chromadb
        from chromadb.config import Settings

        from crewai.rag.chromadb.constants import (
            DEFAULT_DATABASE,
            DEFAULT_STORAGE_PATH,
            DEFAULT_TENANT,
        )
    except Exception:
        return

    try:
        Path(DEFAULT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    try:
        client = chromadb.PersistentClient(
            path=DEFAULT_STORAGE_PATH,
            settings=Settings(
                persist_directory=DEFAULT_STORAGE_PATH,
                allow_reset=True,
                is_persistent=True,
                anonymized_telemetry=False,
            ),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        client.delete_collection("knowledge_crew")
    except Exception:
        pass

    try:
        for old in Path(DEFAULT_STORAGE_PATH).glob(".knowledge_crew_embedder_*"):
            old.unlink(missing_ok=True)
    except OSError:
        pass


_configure_local_rag_embeddings()
_drop_legacy_crew_knowledge_chroma_collection()

# RAG tools index only training corpora — never data/test_review_subset.json
user_rag_tool = JSONSearchTool(
    json_path=str(_PROJECT_ROOT / "data" / "user_subset.json"),
    collection_name="v3_hf_user_data",
    config=rag_config,
)
user_rag_tool.name = "search_user_profile_data"
user_rag_tool.description = (
    "Retrieve a Yelp user's profile: review_count, average_stars, elite years, "
    "compliments, and social stats. Query with user_id or descriptive phrases."
)

item_rag_tool = JSONSearchTool(
    json_path=str(_PROJECT_ROOT / "data" / "item_subset.json"),
    collection_name="v3_hf_item_data",
    config=rag_config,
)
item_rag_tool.name = "search_restaurant_feature_data"
item_rag_tool.description = (
    "Retrieve a business record: name, location, categories, attributes, "
    "aggregate stars, review_count. Query with item_id or business name/location."
)

review_rag_tool = JSONSearchTool(
    json_path=str(_PROJECT_ROOT / "data" / "review_subset.json"),
    collection_name="v3_hf_review_data",
    config=rag_config,
)
review_rag_tool.name = "search_historical_reviews_data"
review_rag_tool.description = (
    "Retrieve historical review text and stars for users or businesses. "
    "Query with user_id, item_id, or topical keywords (food, service, wait, etc.)."
)


def _read_text(rel: str, default: str) -> str:
    path = _PROJECT_ROOT / rel
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return default


_yelp_doc = _read_text(
    "docs/Yelp Data Translation.md",
    "Yelp Data Translation context not available.",
)
_preference_doc = _read_text(
    "knowledge/user_preference.txt",
    "",
)
_crew_knowledge_text = _yelp_doc
if _preference_doc.strip():
    _crew_knowledge_text = (
        _yelp_doc
        + "\n\n---\n\n## Supplementary column / schema notes\n\n"
        + _preference_doc
    )


class ReviewPredictionOutput(BaseModel):
    """Final crew output: predicted stars and synthetic review text."""

    stars: float = Field(
        ge=1.0,
        le=5.0,
        description="Predicted Yelp star rating (1.0–5.0; half-stars allowed).",
    )
    review: str = Field(
        description="First-person Yelp-style review consistent with retrieved evidence.",
    )


@CrewBase
class Week5Lab:
    """Sequential researcher → reporting analyst crew for (user_id, item_id) review prediction."""

    agents: List[BaseAgent]
    tasks: List[Task]

    crew_knowledge = StringKnowledgeSource(content=_crew_knowledge_text)

    @agent
    def prediction_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_researcher"],  # type: ignore[index]
            tools=[user_rag_tool, item_rag_tool, review_rag_tool],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def prediction_reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_reporting_analyst"],  # type: ignore[index]
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def generalist(self) -> Agent:
        return Agent(
            config=self.agents_config["generalist"],  # type: ignore[index]
            verbose=True,
            allow_delegation=False,
        )

    @task
    def prediction_research_task(self) -> Task:
        return Task(
            config=self.tasks_config["prediction_research_task"],  # type: ignore[index]
        )

    @task
    def prediction_report_task(self) -> Task:
        # Task.output_pydantic must be a BaseModel subclass, not CrewAI's YAML wrapper.
        return Task(
            config=self.tasks_config["prediction_report_task"],  # type: ignore[index]
            output_pydantic=ReviewPredictionOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=[self.crew_knowledge],
            embedder=_CREW_KNOWLEDGE_EMBEDDER,
            verbose=True,
        )
