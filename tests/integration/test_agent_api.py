from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from minutes.api.app import create_app
from minutes.evals.runner import run_suite

pytestmark = pytest.mark.integration

QUESTION = "Which sidewalk repair contract items passed, and by what vote?"


def test_demo_step_8_agent(fixture_corpus: str) -> None:
    response = TestClient(create_app()).post("/api/agent", json={"question": QUESTION})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert len(result["steps"]) == 2
    assert [step["tool"] for step in result["steps"]] == ["search", "sql"]
    assert "RESOLUTION 2024-07 passed 4-1" in result["answer"]
    assert result["max_steps"] == 8


@pytest.mark.parametrize("question", ["", "x" * 501])
def test_agent_validation_error_is_422(question: str) -> None:
    response = TestClient(create_app()).post("/api/agent", json={"question": question})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_agent_section_reports_completion_over_five_runs(
    fixture_corpus: str, tmp_path: Path
) -> None:
    results = run_suite(
        "full",
        controls=True,
        labels_dir=Path("tests/fixtures/labels"),
        out_path=tmp_path / "results.json",
        only={"agent"},
    )
    agent_result = cast(dict[str, object], results["agent"])
    completion = cast(dict[str, object], agent_result["completion"])
    assert completion["runs"] == 5
