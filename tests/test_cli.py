"""Tests for the CLI entrypoint."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from docuresearch.cli import main


def test_cli_mock_run_exits_zero(capsys: object) -> None:
    exit_code = main(["--mock"])
    assert exit_code == 0


def test_cli_requires_topic_without_mock() -> None:
    try:
        main([])
        raised = False
    except SystemExit as exc:
        raised = True
        assert exc.code != 0
    assert raised, "argparse should exit non-zero when --topic is missing and --mock is not set"


def test_cli_writes_json_output(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"
    exit_code = main(["--mock", "--output", str(output_path)])
    assert exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert '"topic"' in content


def test_cli_research_only_skips_script_and_prints_resume_hint(capsys: pytest.CaptureFixture) -> None:
    exit_code = main(["--mock", "--research-only"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "--- FINAL SCRIPT ---" not in out
    assert "--- HOOK OPTIONS ---" not in out
    assert "Resume with: --resume " in out


def test_cli_resume_script_only_completes_the_run(capsys: pytest.CaptureFixture) -> None:
    exit_code = main(["--mock", "--research-only"])
    assert exit_code == 0
    out = capsys.readouterr().out
    match = re.search(r"Research ID: (\S+)", out)
    assert match is not None
    research_id = match.group(1)

    exit_code = main(["--resume", research_id, "--script-only"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "--- FINAL SCRIPT ---" in out
    assert f"Research ID: {research_id}" in out


def test_cli_resume_unknown_id_returns_error() -> None:
    exit_code = main(["--resume", "RUN-does-not-exist"])
    assert exit_code == 1


def test_cli_research_only_with_resume_is_rejected() -> None:
    try:
        main(["--research-only", "--resume", "RUN-anything"])
        raised = False
    except SystemExit as exc:
        raised = True
        assert exc.code != 0
    assert raised, "--research-only and --resume together should be rejected by argparse.error"


def test_cli_list_runs_shows_previously_created_run(capsys: pytest.CaptureFixture) -> None:
    exit_code = main(["--mock", "--research-only"])
    assert exit_code == 0
    out = capsys.readouterr().out
    match = re.search(r"Research ID: (\S+)", out)
    assert match is not None
    research_id = match.group(1)

    exit_code = main(["--list-runs"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert research_id in out
