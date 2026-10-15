import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from minutes import cli, db
from minutes.errors import GateStaleError
from minutes.evals import gate, runner

RECORDINGS = ("embeddings.npz", "rerank.jsonl.gz", "snapshot.jsonl.gz", "llm.jsonl.gz")


@pytest.mark.parametrize("missing", RECORDINGS)
def test_missing_recording_raises_stale_error_before_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    for name in RECORDINGS:
        if name != missing:
            (tmp_path / name).touch()
    reset = MagicMock()
    monkeypatch.setattr(db, "reset", reset)
    monkeypatch.setattr(cli, "configure_logging", MagicMock())
    monkeypatch.setattr(cli, "configure_tracing", MagicMock())
    with pytest.raises(GateStaleError) as caught:
        gate.run_gate(recording_dir=tmp_path, results_path=tmp_path / "results.json")
    assert str(caught.value) == f"gate recording is stale: missing {missing}"
    monkeypatch.setattr(cli, "GATE_RECORDING_DIR", tmp_path)
    result = CliRunner().invoke(cli.app, ["eval", "gate"])
    assert result.exit_code == 3
    assert f"gate recording is stale: missing {missing}" in result.output
    reset.assert_not_called()


def test_gate_discards_previous_results_before_running_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in RECORDINGS:
        (tmp_path / name).touch()
    results = tmp_path / "results.json"
    previous = {"answers": {"final": {"answer_accuracy": {"median": 1.0}}}}
    results.write_text(json.dumps(previous), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"metrics": {"answer_accuracy": 1.0}}), encoding="utf-8")
    monkeypatch.setattr(db, "reset", MagicMock())
    monkeypatch.setattr(db, "connect", MagicMock())
    monkeypatch.setattr(gate, "load_snapshot", MagicMock())
    monkeypatch.setattr(gate, "gate_document_ids_from_snapshot", lambda path: [])

    def run_suite(*args: object, **kwargs: object) -> dict[str, object]:
        assert not results.exists()
        return runner.write_results(results, {"segmentation": {"overall": {"exact_rate": 1}}})

    monkeypatch.setattr(runner, "run_suite", run_suite)
    assert (
        gate.run_gate(
            recording_dir=tmp_path,
            baseline_path=baseline,
            labels_dir=tmp_path,
            results_path=results,
        )
        == 1
    )
    assert "answers" not in json.loads(results.read_text(encoding="utf-8"))
