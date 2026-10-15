import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest
from typer.testing import CliRunner

from minutes import db
from minutes.cli import app
from minutes.config import get_settings
from minutes.errors import ConfigError, GateStaleError
from minutes.evals import gate, runner
from minutes.ingest import pipeline

pytestmark = pytest.mark.integration
FIXTURE_LABELS = Path("tests/fixtures/labels")


@pytest.fixture(autouse=True)
def clear_cached_settings() -> Iterator[None]:
    yield
    get_settings.cache_clear()


@dataclass(frozen=True)
class GateFiles:
    recording: Path
    labels: Path
    results: Path
    baseline: Path


@pytest.fixture(scope="module")
def gate_files(
    fixture_corpus: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[GateFiles]:
    # the gate resets its database, so it gets its own instead of the shared fixture one;
    # the snapshot is taken from the shared fixture database
    url = urlsplit(fixture_corpus)._replace(path="/minutes_gate_test").geturl()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MINUTES_TEST_DATABASE_URL", url)
        get_settings.cache_clear()
        yield _record(tmp_path_factory, fixture_corpus)
    get_settings.cache_clear()


def _record(tmp_path_factory: pytest.TempPathFactory, source_url: str) -> GateFiles:
    root = tmp_path_factory.mktemp("gate")
    recording = root / "recording"
    labels = root / "labels"
    labels.mkdir()
    for path in FIXTURE_LABELS.glob("*.jsonl"):
        shutil.copyfile(path, labels / path.name)
    (labels / "manifest.json").write_text('{"frozen":true}\n', encoding="utf-8")
    results = root / "gate_results.json"
    baseline = root / "baseline.json"
    gate.record(
        recording,
        labels_dir=labels,
        source_url=source_url,
        results_path=results,
    )
    return GateFiles(recording, labels, results, baseline)


def test_snapshot_round_trip_restores_documents_and_units(
    gate_files: GateFiles, fixture_corpus: str
) -> None:
    ids = gate.gate_document_ids_from_snapshot(gate_files.recording / "snapshot.jsonl.gz")
    with db.connect(fixture_corpus) as conn:
        expected_documents = gate._rows(
            conn, "SELECT * FROM documents WHERE id = ANY(%s) ORDER BY id", (ids,)
        )
        expected_units = gate._rows(
            conn,
            "SELECT * FROM units WHERE document_id = ANY(%s) ORDER BY document_id, unit_index",
            (ids,),
        )
    for row in expected_units:
        row.pop("boxes")
    db.reset(get_settings().test_database_url)
    with db.connect(get_settings().test_database_url) as conn:
        gate.load_snapshot(conn, gate_files.recording)
        documents = gate._rows(conn, "SELECT * FROM documents ORDER BY id", ())
        units = gate._rows(conn, "SELECT * FROM units ORDER BY document_id, unit_index", ())
    for row in units:
        row.pop("boxes")
    assert documents == expected_documents
    assert units == expected_units


def test_snapshot_bytes_are_reproducible(
    gate_files: GateFiles, fixture_corpus: str, tmp_path: Path
) -> None:
    with db.connect(fixture_corpus) as conn:
        path = gate.build_snapshot(
            conn,
            tmp_path / "second",
            labels=runner.load_labels(gate_files.labels),
        )
    first_hash = hashlib.sha256(
        (gate_files.recording / "snapshot.jsonl.gz").read_bytes()
    ).hexdigest()
    second_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert first_hash == second_hash


def test_record_then_gate_passes_in_replay_mode(
    gate_files: GateFiles, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("MINUTES_LLM_MODE", "replay")
    monkeypatch.setenv("MINUTES_MODELS_MODE", "replay")
    get_settings.cache_clear()
    assert (
        gate.run_gate(
            recording_dir=gate_files.recording,
            baseline_path=gate_files.baseline,
            labels_dir=gate_files.labels,
            results_path=gate_files.results,
        )
        == 0
    )
    assert "gate passed" in capsys.readouterr().out


def test_replay_makes_no_model_call_and_imports_no_gpu_library(
    gate_files: GateFiles,
) -> None:
    code = (
        "import json, sys; "
        "from unittest.mock import patch; "
        "from pathlib import Path; "
        "from minutes.models import embedder, reranker; "
        "from minutes.evals.gate import run_gate; "
        "patch('httpx.Client.send', side_effect=AssertionError('http request sent')).start(); "
        "patch.object(embedder, '_load_model', "
        "side_effect=AssertionError('embed model loaded')).start(); "
        "patch.object(reranker, '_load_model', "
        "side_effect=AssertionError('reranker loaded')).start(); "
        f"result = run_gate(recording_dir=Path({str(gate_files.recording)!r}), "
        f"baseline_path=Path({str(gate_files.baseline)!r}), "
        f"labels_dir=Path({str(gate_files.labels)!r}), "
        f"results_path=Path({str(gate_files.results)!r})); "
        "print(json.dumps({'result': result, 'gpu_modules': "
        "sorted({'torch', 'sentence_transformers', 'doctr'} & sys.modules.keys())}))"
    )
    env = os.environ.copy()
    env.update(
        {
            "MINUTES_LLM_MODE": "replay",
            "MINUTES_MODELS_MODE": "replay",
            "MINUTES_CORPUS": "fixture",
            "MINUTES_LLM_BASE_URL": "",
            "MINUTES_LLM_API_KEY": "",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    # the subprocess recreated the gate database, so the pooled connections here are dead
    stale = db._pools.pop(get_settings().test_database_url, None)
    if stale is not None:
        stale.close()
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.splitlines()[-1])
    assert output["result"] == 0
    assert output["gpu_modules"] == []


def test_missing_embedding_key_raises_stale_error(
    gate_files: GateFiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINUTES_LLM_MODE", "replay")
    monkeypatch.setenv("MINUTES_MODELS_MODE", "replay")
    get_settings.cache_clear()
    path = gate_files.recording / "embeddings.npz"
    original = path.read_bytes()
    try:
        with db.connect(get_settings().test_database_url) as conn:
            first_row = conn.execute(
                "SELECT embed_text FROM chunks ORDER BY document_id, id LIMIT 1"
            ).fetchone()
        assert first_row is not None
        first_text = first_row[0]
        missing_key = (
            hashlib.sha256(f"embed\n{gate.ReplayEmbedder.model_id}\npassage\n{first_text}".encode())
            .hexdigest()
            .encode("ascii")
        )
        with np.load(path) as recording:
            keep = recording["keys"] != missing_key
            keys = recording["keys"][keep]
            vectors = recording["vectors"][keep]
        np.savez_compressed(path, keys=keys, vectors=vectors)
        with pytest.raises(GateStaleError, match="gate recording is stale: missing embed key"):
            gate.run_gate(
                recording_dir=gate_files.recording,
                baseline_path=gate_files.baseline,
                labels_dir=gate_files.labels,
                results_path=gate_files.results,
            )
        from minutes import cli

        monkeypatch.setattr(cli, "GATE_RECORDING_DIR", gate_files.recording)
        monkeypatch.setattr(cli, "GATE_BASELINE_FILE", gate_files.baseline)
        monkeypatch.setattr(cli, "GATE_RESULTS_FILE", gate_files.results)
        monkeypatch.setattr(cli, "LABELS_DIR", gate_files.labels)
        result = CliRunner().invoke(app, ["eval", "gate"])
        assert result.exit_code == 3
    finally:
        path.write_bytes(original)


def test_regression_makes_gate_fail(
    gate_files: GateFiles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINUTES_LLM_MODE", "replay")
    monkeypatch.setenv("MINUTES_MODELS_MODE", "replay")
    get_settings.cache_clear()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"metrics": {"segmentation_exact_rate": 1.0}}), encoding="utf-8")
    monkeypatch.setitem(
        pipeline.STAGE_FUNCTIONS,
        "segment",
        lambda conn, document_id: {"items": 0},
    )
    from minutes import cli

    monkeypatch.setattr(cli, "GATE_RECORDING_DIR", gate_files.recording)
    monkeypatch.setattr(cli, "GATE_BASELINE_FILE", baseline)
    monkeypatch.setattr(cli, "GATE_RESULTS_FILE", gate_files.results)
    monkeypatch.setattr(cli, "LABELS_DIR", gate_files.labels)
    result = CliRunner().invoke(app, ["eval", "gate"])
    assert result.exit_code == 1
    assert any(line.endswith("FAIL") for line in result.stdout.splitlines())
    assert result.stdout.rstrip().endswith("gate failed")


def test_gate_requires_frozen_manifest(fixture_corpus: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"^label files are not frozen$"):
        runner.run_suite(
            "gate",
            controls=True,
            labels_dir=tmp_path,
            out_path=tmp_path / "results.json",
        )
