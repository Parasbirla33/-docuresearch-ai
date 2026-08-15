"""Tests for storage/repository.py's ResearchRunRepository (Phase 7)."""

from __future__ import annotations

import time
from pathlib import Path

from docuresearch.models.state import new_initial_state
from docuresearch.storage.models import RunStatus
from docuresearch.storage.repository import ResearchRunRepository


def _repo(tmp_path: Path) -> ResearchRunRepository:
    return ResearchRunRepository(database_url=f"sqlite:///{(tmp_path / 'runs.db').as_posix()}")


def test_load_unknown_id_returns_none(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert repo.load("RUN-does-not-exist") is None
    assert repo.get_status("RUN-does-not-exist") is None


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    state = new_initial_state(research_id="RUN-1", topic="Topic A", mock_mode=True)

    repo.save_checkpoint(state, status=RunStatus.RESEARCH_COMPLETE)
    loaded = repo.load("RUN-1")

    assert loaded is not None
    assert loaded["research_id"] == "RUN-1"
    assert loaded["topic"] == "Topic A"
    assert repo.get_status("RUN-1") == RunStatus.RESEARCH_COMPLETE.value


def test_save_checkpoint_upserts_and_advances_status(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    state = new_initial_state(research_id="RUN-2", topic="Topic B")

    repo.save_checkpoint(state, status=RunStatus.RESEARCH_COMPLETE)
    repo.save_checkpoint(state, status=RunStatus.COMPLETE)

    assert repo.get_status("RUN-2") == RunStatus.COMPLETE.value
    assert len(repo.list_runs()) == 1  # same id -> one row, not two


def test_list_runs_most_recently_updated_first(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.save_checkpoint(
        new_initial_state(research_id="RUN-old", topic="Old"), status=RunStatus.RESEARCH_COMPLETE
    )
    time.sleep(0.01)  # guarantee a distinct updated_at from the first save
    repo.save_checkpoint(
        new_initial_state(research_id="RUN-new", topic="New"), status=RunStatus.RESEARCH_COMPLETE
    )

    runs = repo.list_runs()
    ids = [r.research_id for r in runs]
    assert "RUN-old" in ids
    assert "RUN-new" in ids
    assert ids.index("RUN-new") < ids.index("RUN-old")


def test_list_runs_respects_limit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    for i in range(5):
        repo.save_checkpoint(
            new_initial_state(research_id=f"RUN-{i}", topic=f"Topic {i}"),
            status=RunStatus.RESEARCH_COMPLETE,
        )

    assert len(repo.list_runs(limit=2)) == 2
