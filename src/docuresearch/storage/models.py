"""SQLAlchemy ORM model for persisted research runs (Phase 7).

One JSON blob column holds the full serialized `DocuResearchState` snapshot -
consistent with treating the state dict as the single source of truth, and
avoids a large normalized schema that a future API layer may want to design
differently anyway.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, Boolean, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class RunStatus(StrEnum):
    """Checkpoint granularity: phase boundaries, not per-node.

    A run that crashes mid-phase leaves no row for its id - resuming is only
    meaningful at these two boundaries.
    """

    RESEARCH_COMPLETE = "research_complete"
    COMPLETE = "complete"


class Base(DeclarativeBase):
    pass


class ResearchRunORM(Base):
    """A persisted checkpoint of one research run's state."""

    __tablename__ = "research_runs"

    research_id: Mapped[str] = mapped_column(String, primary_key=True)
    topic: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    mock_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    state: Mapped[dict] = mapped_column(JSON)
