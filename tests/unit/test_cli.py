import json
from pathlib import Path

import pytest
import typer

from minutes import cli
from minutes.evals import grader, runner
from minutes.evals.grader import GraderScore, GraderValidation


@pytest.mark.parametrize("name", ["answer", "support"])
@pytest.mark.parametrize("accuracy,kappa", [(None, 1.0), (1.0, None), (None, None)])
def test_validate_grader_null_score_fails_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    accuracy: float | None,
    kappa: float | None,
) -> None:
    scores = {"answer": GraderScore(1.0, 1.0), "support": GraderScore(1.0, 1.0)}
    scores[name] = GraderScore(accuracy, kappa)
    validation = GraderValidation(scores["answer"], scores["support"])
    output = tmp_path / "results.json"
    monkeypatch.setattr(cli, "RESULTS_FILE", output)
    monkeypatch.setattr(runner, "load_labels", lambda path: [])
    monkeypatch.setattr(grader, "validate", lambda labels: validation)
    with pytest.raises(typer.Exit) as caught:
        cli.eval_validate_grader()
    assert caught.value.exit_code == 1
    saved = json.loads(output.read_text(encoding="utf-8"))["grader"][name]
    assert saved == {"accuracy": accuracy, "kappa": kappa}
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith(name))
    if accuracy is None:
        assert "accuracy=null" in line
    if kappa is None:
        assert "kappa=null" in line
