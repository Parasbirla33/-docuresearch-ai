"""SQLAlchemy-backed persistence for research runs (Phase 7).

Follows the same DI convention as `llm/factory.py`: an optional parameter,
defaulting to the settings singleton, so tests can point this at a throwaway
database without touching `data/docuresearch.db`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from docuresearch.config.settings import get_settings
from docuresearch.models.state import DocuResearchState, deserialize_state, serialize_state
from docuresearch.storage.models import Base, ResearchRunORM, RunStatus


@dataclass
class RunSummary:
    """Lightweight listing row - no full state payload."""

    research_id: str
    topic: str
    status: str
    mock_mode: bool
    updated_at: datetime


class ResearchRunRepository:
    """CRUD over persisted `DocuResearchState` checkpoints."""

    def __init__(self, database_url: str | None = None) -> None:
        url = database_url or get_settings().database_url
        self._engine = create_engine(url)
        Base.metadata.create_all(self._engine)
        self._session_factory: sessionmaker[Session] = sessionmaker(bind=self._engine)

    def save_checkpoint(self, state: DocuResearchState, *, status: RunStatus) -> None:
        """Upsert the current state as a checkpoint for `state["research_id"]`."""
        research_id = state["research_id"]
        payload = serialize_state(state)
        now = datetime.now(UTC)

        with self._session_factory() as session:
            row = session.get(ResearchRunORM, research_id)
            if row is None:
                row = ResearchRunORM(research_id=research_id, created_at=now)
                session.add(row)
            row.topic = state.get("topic", "")
            row.status = status.value
            row.mock_mode = bool(state.get("mock_mode", False))
            row.updated_at = now
            row.state = payload
            session.commit()

    def load(self, research_id: str) -> DocuResearchState | None:
        """Reconstruct the persisted state for `research_id`, or None if unknown."""
        with self._session_factory() as session:
            row = session.get(ResearchRunORM, research_id)
            if row is None:
                return None
            return deserialize_state(row.state)

    def get_status(self, research_id: str) -> str | None:
        with self._session_factory() as session:
            row = session.get(ResearchRunORM, research_id)
            return row.status if row is not None else None

    def list_runs(self, limit: int = 20) -> list[RunSummary]:
        """Most-recently-updated runs first, id/topic/status/updated_at only."""
        with self._session_factory() as session:
            stmt = (
                select(ResearchRunORM)
                .order_by(ResearchRunORM.updated_at.desc())
                .limit(limit)
            )
            rows = session.scalars(stmt).all()
            return [
                RunSummary(
                    research_id=row.research_id,
                    topic=row.topic,
                    status=row.status,
                    mock_mode=row.mock_mode,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]
