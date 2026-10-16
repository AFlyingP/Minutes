import gzip
import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from math import isclose
from pathlib import Path
from typing import Any, cast

import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from minutes import db
from minutes.config import CONFIG_DIR, ROOT, get_settings, load_models
from minutes.errors import ConfigError, GateStaleError, LLMReplayMiss
from minutes.evals import runner
from minutes.ingest import pipeline
from minutes.labels.schema import parse_payload
from minutes.llm import Budget, ChatResult, LLMClient
from minutes.llm.cache import DiskCache, recording_signature
from minutes.llm.client import request_cache_key
from minutes.models import Embedder, Reranker

SNAPSHOT_TABLES = ("cities", "meetings", "documents", "units")
GATE_STAGES: tuple[str, ...] = ("segment", "chunk", "embed", "extract")
REPLAY_DIR = ROOT / "eval" / "gate"
GATED: dict[str, tuple[str, str, float]] = {
    "recall_at_10": ("retrieval.item.hybrid_rerank.recall_at_10", "lower", 0.02),
    "answer_accuracy": ("answers.final.answer_accuracy.median", "lower", 0.03),
    "faithfulness": ("answers.final.faithfulness.median", "lower", 0.03),
    "decline_accuracy": ("answers.final.decline_accuracy.median", "lower", 0.084),
    "extraction_accuracy": ("extraction.overall.median", "lower", 0.02),
    "segmentation_exact_rate": ("segmentation.overall.exact_rate", "lower", 0.05),
    "agent_completion": ("agent.completion.median", "lower", 0.05),
    "cost_per_query_usd": ("cost.controls_on.cost_per_query_usd", "cost", 0.10),
}


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        for row in rows:
            compressed.write(
                (json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode(
                    "utf-8"
                )
            )


def gate_results_path() -> Path:
    data_dir = get_settings().data_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return data_dir / "gate_results.json"


def gate_document_ids(conn: psycopg.Connection[TupleRow], labels: runner.Labels) -> list[str]:
    document_ids: set[str] = set()
    agenda_meetings: set[str] = set()
    for label in labels:
        label_type = label.type
        payload = parse_payload(label_type, label.payload)
        data = payload.model_dump()
        if label_type == "question":
            document_ids.update(passage["document_id"] for passage in data.get("passages", []))
        elif label_type == "extraction":
            document_ids.add(data["passage"]["document_id"])
        elif label_type == "agent_task":
            document_ids.update(data["document_ids"])
        elif label_type == "agenda_count":
            agenda_meetings.add(data["meeting_id"])
    if agenda_meetings:
        rows = conn.execute(
            "SELECT id FROM documents WHERE meeting_id = ANY(%s) AND kind = 'agenda' ORDER BY id",
            (sorted(agenda_meetings),),
        ).fetchall()
        document_ids.update(row[0] for row in rows)
    subset = json.loads((CONFIG_DIR / "dev_subset.json").read_text(encoding="utf-8"))
    document_ids.update(subset["documents"])
    return sorted(document_ids)


def _rows(
    conn: psycopg.Connection[TupleRow],
    query: str,
    params: tuple[object, ...],
) -> list[dict[str, object]]:
    cursor = conn.execute(query, params)
    assert cursor.description is not None
    columns = [column.name for column in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def build_snapshot(
    conn: psycopg.Connection[TupleRow],
    out_dir: Path | None = None,
    *,
    labels: runner.Labels | None = None,
) -> Path:
    out_dir = out_dir or ROOT / "eval" / "gate"
    labels = labels if labels is not None else runner.load_labels(ROOT / "eval" / "labels")
    document_ids = gate_document_ids(conn, labels)
    doc_rows = _rows(
        conn,
        "SELECT * FROM documents WHERE id = ANY(%s) ORDER BY id",
        (document_ids,),
    )
    meeting_ids = sorted({cast(str, row["meeting_id"]) for row in doc_rows})
    city_ids = sorted({cast(str, row["city_id"]) for row in doc_rows})
    rows: list[dict[str, object]] = []
    for table, query, ids in (
        ("cities", "SELECT * FROM cities WHERE id = ANY(%s) ORDER BY id", city_ids),
        ("meetings", "SELECT * FROM meetings WHERE id = ANY(%s) ORDER BY id", meeting_ids),
    ):
        rows.extend({"table": table, "row": row} for row in _rows(conn, query, (ids,)))
    rows.extend({"table": "documents", "row": row} for row in doc_rows)
    unit_rows = _rows(
        conn,
        "SELECT * FROM units WHERE document_id = ANY(%s) ORDER BY document_id, unit_index",
        (document_ids,),
    )
    for row in unit_rows:
        row.pop("boxes", None)
        rows.append({"table": "units", "row": row})
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "snapshot.jsonl.gz"
    _write_jsonl_gzip(path, rows)
    return path


def load_snapshot(conn: psycopg.Connection[TupleRow], in_dir: Path) -> None:
    rows = _read_jsonl_gzip(in_dir / "snapshot.jsonl.gz")
    for table in SNAPSHOT_TABLES:
        selected = [cast(dict[str, object], row["row"]) for row in rows if row["table"] == table]
        if not selected:
            continue
        columns = list(selected[0])
        statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        )
        with conn.cursor() as cursor:
            cursor.executemany(statement, [[row[column] for column in columns] for row in selected])


class Recorder:
    def __init__(self) -> None:
        self.llm: dict[str, dict[str, Any]] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        self.rerank: dict[str, float] = {}

    def embedder(self, model: Embedder) -> Embedder:
        return RecordingEmbedder(model, self)

    def reranker(self, model: Reranker) -> Reranker:
        return RecordingReranker(model, self)

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl_gzip(
            out_dir / "llm.jsonl.gz",
            [{"key": key, "value": self.llm[key]} for key in sorted(self.llm)],
        )
        keys = sorted(self.embeddings)
        vectors = (
            np.stack([self.embeddings[key] for key in keys]).astype(np.float16)
            if keys
            else np.empty((0, 768), dtype=np.float16)
        )
        np.savez_compressed(
            out_dir / "embeddings.npz",
            keys=np.asarray(keys, dtype="S64"),
            vectors=vectors,
        )
        _write_jsonl_gzip(
            out_dir / "rerank.jsonl.gz",
            [{"key": key, "score": self.rerank[key]} for key in sorted(self.rerank)],
        )


@lru_cache(maxsize=8)
def _embedding_values(path: Path, mtime_ns: int, size: int) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {
            key.decode("ascii"): vector.astype(np.float32)
            for key, vector in zip(values["keys"], values["vectors"], strict=True)
        }


@lru_cache(maxsize=8)
def _rerank_values(path: Path, mtime_ns: int, size: int) -> dict[str, float]:
    return {row["key"]: float(row["score"]) for row in _read_jsonl_gzip(path)}


class ReplayEmbedder:
    model_id = "BAAI/bge-base-en-v1.5"
    dim = 768

    def __init__(self, path: Path = ROOT / "eval" / "gate" / "embeddings.npz") -> None:
        self.values = _embedding_values(*recording_signature(path))

    def _embed(self, kind: str, text: str) -> np.ndarray:
        key = hashlib.sha256(f"embed\n{self.model_id}\n{kind}\n{text}".encode()).hexdigest()
        try:
            return self.values[key]
        except KeyError as err:
            raise GateStaleError(
                f"gate recording is stale: missing embed key for {text[:40]}"
            ) from err

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed("passage", text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed("query", text)


class ReplayReranker:
    model_id = "BAAI/bge-reranker-v2-m3"

    def __init__(self, path: Path = ROOT / "eval" / "gate" / "rerank.jsonl.gz") -> None:
        self.values = _rerank_values(*recording_signature(path))

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = []
        for passage in passages:
            key = hashlib.sha256(
                f"rerank\n{self.model_id}\n{query}\n{passage}".encode()
            ).hexdigest()
            try:
                scores.append(self.values[key])
            except KeyError as err:
                raise GateStaleError(
                    f"gate recording is stale: missing rerank key for {passage[:40]}"
                ) from err
        return scores


def _nested(data: object, path: str) -> object:
    value = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def extract_gated(results: Mapping[str, object]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name, (path, _, _) in GATED.items():
        value = _nested(results, path)
        values[name] = float(value) if isinstance(value, (int, float)) else None
    return values


@dataclass(frozen=True)
class GateFailure:
    metric: str
    baseline: float
    current: float | None
    tolerance: float
    source: str


def compare(results: Mapping[str, object], baseline: Mapping[str, float]) -> list[GateFailure]:
    current = extract_gated(results)
    failures = []
    for name, baseline_value in baseline.items():
        if name not in GATED:
            continue
        path, direction, tolerance = GATED[name]
        value = current[name]
        failed = value is None
        if value is not None:
            if direction == "cost":
                threshold = baseline_value * (1 + tolerance)
                failed = value > threshold and not isclose(value, threshold, abs_tol=1e-12)
            else:
                threshold = baseline_value - tolerance
                failed = value < threshold and not isclose(value, threshold, abs_tol=1e-12)
        if failed:
            failures.append(GateFailure(name, baseline_value, value, tolerance, path))
    return failures


def _baseline_metrics(results: dict[str, object]) -> dict[str, float]:
    return {name: value for name, value in extract_gated(results).items() if value is not None}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_now() -> datetime:
    return datetime.now(UTC)


def _record_manifest(
    out_dir: Path,
    recorder: Recorder,
    snapshot: Path,
    now: Callable[[], datetime] = _default_now,
) -> None:
    files = [
        snapshot,
        out_dir / "llm.jsonl.gz",
        out_dir / "embeddings.npz",
        out_dir / "rerank.jsonl.gz",
    ]
    manifest = {
        "recorded_utc": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": runner._git_commit(),
        "stage_versions": {stage: pipeline.STAGE_VERSIONS[stage] for stage in GATE_STAGES},
        "files": {path.name: _sha256(path) for path in files},
        "counts": {
            "documents": len(gate_document_ids_from_snapshot(snapshot)),
            "units": sum(1 for row in _read_jsonl_gzip(snapshot) if row["table"] == "units"),
            "llm": len(recorder.llm),
            "embeddings": len(recorder.embeddings),
            "rerank": len(recorder.rerank),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _drop_fixed_chunks(conn: psycopg.Connection[TupleRow], document_id: str) -> None:
    conn.execute(
        "DELETE FROM chunks WHERE document_id = %s AND chunker = 'fixed' "
        "AND EXISTS (SELECT 1 FROM chunks WHERE document_id = %s AND chunker = 'item')",
        (document_id, document_id),
    )


def gate_document_ids_from_snapshot(path: Path) -> list[str]:
    return [row["row"]["id"] for row in _read_jsonl_gzip(path) if row["table"] == "documents"]


class RecordingEmbedder:
    def __init__(self, model: Embedder, recorder: Recorder) -> None:
        self.model = model
        self.recorder = recorder
        # the replay models carry the real model IDs, so recorded keys must use them too
        self.model_id = ReplayEmbedder.model_id
        self.dim = model.dim

    def _keep(self, kind: str, text: str, vector: np.ndarray) -> None:
        key = hashlib.sha256(f"embed\n{self.model_id}\n{kind}\n{text}".encode()).hexdigest()
        self.recorder.embeddings[key] = np.asarray(vector)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.embed_passages(texts)
        for text, vector in zip(texts, vectors, strict=True):
            self._keep("passage", text, vector)
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.embed_query(text)
        self._keep("query", text, vector)
        return vector


class RecordingReranker:
    def __init__(self, model: Reranker, recorder: Recorder) -> None:
        self.model = model
        self.recorder = recorder
        self.model_id = ReplayReranker.model_id

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores = self.model.score(query, passages)
        for passage, score in zip(passages, scores, strict=True):
            key = hashlib.sha256(
                f"rerank\n{self.model_id}\n{query}\n{passage}".encode()
            ).hexdigest()
            self.recorder.rerank[key] = float(score)
        return scores


class RecordingCache(DiskCache):
    """The development LLM cache, copying every entry read or written into the recorder."""

    def __init__(self, root: Path, recorder: Recorder) -> None:
        super().__init__(root)
        self.recorder = recorder

    def get(self, key: str) -> dict[str, Any] | None:
        value = super().get(key)
        if value is not None:
            self.recorder.llm[key] = value
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        super().put(key, value)
        self.recorder.llm[key] = value


class RecordingStubClient:
    def __init__(self, client: LLMClient, recorder: Recorder) -> None:
        self.client = client
        self.recorder = recorder

    def chat(
        self,
        *,
        purpose: str,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None = None,
        schema_name: str = "output",
        model: str | None = None,
        sample_index: int = 0,
        budget: Budget | None = None,
    ) -> ChatResult:
        result = self.client.chat(
            purpose=purpose,
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            model=model,
            sample_index=sample_index,
            budget=budget,
        )
        purpose_config = load_models().purposes[purpose]
        self.recorder.llm[
            request_cache_key(
                result.model,
                messages,
                purpose_config.max_tokens,
                purpose_config.reasoning_effort,
                schema,
                schema_name,
                sample_index,
            )
        ] = {
            "text": result.text,
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
        }
        return result


@contextmanager
def _recording(recorder: Recorder) -> Iterator[None]:
    from minutes import llm, models

    previous_get_client = llm.get_client

    def recording_get_client() -> LLMClient:
        client = previous_get_client()
        if get_settings().llm_mode == "stub":
            return RecordingStubClient(client, recorder)
        return client

    models.recorder = recorder
    llm.cache_recorder = lambda root: RecordingCache(root, recorder)
    llm.get_client = recording_get_client
    try:
        yield
    finally:
        llm.get_client = previous_get_client
        models.recorder = None
        llm.cache_recorder = None


def record(
    out_dir: Path | None = None,
    *,
    labels_dir: Path | None = None,
    source_url: str | None = None,
    results_path: Path | None = None,
) -> Recorder:
    out_dir = out_dir or ROOT / "eval" / "gate"
    labels_dir = labels_dir or ROOT / "eval" / "labels"
    results_path = results_path or gate_results_path()
    settings = get_settings()
    labels = runner.load_labels(labels_dir)
    recorder = Recorder()
    with db.connect(source_url or settings.database_url) as conn:
        snapshot = build_snapshot(conn, out_dir, labels=labels)
    db.reset(settings.test_database_url)
    with _settings_override(
        {
            "MINUTES_CORPUS": "fixture",
        }
    ):
        with db.connect(settings.test_database_url) as conn:
            load_snapshot(conn, out_dir)
            ids = gate_document_ids_from_snapshot(snapshot)
            with _recording(recorder):
                for stage in GATE_STAGES:
                    for document_id in ids:
                        pipeline.run_stage(conn, stage, document_id, "fixture")
                        if stage == "chunk":
                            _drop_fixed_chunks(conn, document_id)
        with _recording(recorder):
            runner.run_suite("gate", controls=True, labels_dir=labels_dir, out_path=results_path)
    recorder.write(out_dir)
    _record_manifest(out_dir, recorder, snapshot)
    return recorder


@contextmanager
def _settings_override(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@contextmanager
def _replay_directory(path: Path) -> Iterator[None]:
    from minutes import llm

    global REPLAY_DIR
    previous_gate, previous_llm = REPLAY_DIR, llm.REPLAY_DIR
    REPLAY_DIR = path
    llm.REPLAY_DIR = path
    try:
        yield
    finally:
        REPLAY_DIR = previous_gate
        llm.REPLAY_DIR = previous_llm


def run_gate(
    *,
    recording_dir: Path | None = None,
    baseline_path: Path | None = None,
    labels_dir: Path | None = None,
    results_path: Path | None = None,
) -> int:
    settings = get_settings()
    recording_dir = recording_dir or ROOT / "eval" / "gate"
    baseline_path = baseline_path or ROOT / "eval" / "baseline.json"
    labels_dir = labels_dir or ROOT / "eval" / "labels"
    results_path = results_path or gate_results_path()
    for name in ("snapshot.jsonl.gz", "embeddings.npz", "rerank.jsonl.gz", "llm.jsonl.gz"):
        if not (recording_dir / name).is_file():
            raise GateStaleError(f"gate recording is stale: missing {name}")
    db.reset(settings.test_database_url)
    with (
        _replay_directory(recording_dir),
        _settings_override(
            {
                "MINUTES_LLM_MODE": "replay",
                "MINUTES_MODELS_MODE": "replay",
                "MINUTES_CORPUS": "fixture",
                "MINUTES_LLM_BASE_URL": "",
                "MINUTES_LLM_API_KEY": "",
            }
        ),
    ):
        try:
            with db.connect(settings.test_database_url) as conn:
                load_snapshot(conn, recording_dir)
                ids = gate_document_ids_from_snapshot(recording_dir / "snapshot.jsonl.gz")
                for stage in GATE_STAGES:
                    for document_id in ids:
                        pipeline.run_stage(conn, stage, document_id, "fixture")
                        if stage == "chunk":
                            _drop_fixed_chunks(conn, document_id)
            results_path.unlink(missing_ok=True)
            results = runner.run_suite(
                "gate", controls=True, labels_dir=labels_dir, out_path=results_path
            )
        except LLMReplayMiss as err:
            key = str(err).rsplit(" ", 1)[-1][:40]
            raise GateStaleError(f"gate recording is stale: missing llm key for {key}") from err
    baseline: dict[str, float] = {}
    if baseline_path.exists():
        baseline_file = cast(
            dict[str, object], json.loads(baseline_path.read_text(encoding="utf-8"))
        )
        baseline = cast(dict[str, float], baseline_file.get("metrics", {}))
    failures = compare(results, baseline)
    current = extract_gated(results)
    for name, (_, direction, tolerance) in GATED.items():
        value = current[name]
        baseline_value = baseline.get(name)
        status = "FAIL" if any(item.metric == name for item in failures) else "ok"
        shown = "n/a" if value is None else f"{value:g}"
        prior = "n/a" if baseline_value is None else f"{baseline_value:g}"
        tolerance_text = f"{tolerance * 100:g}%" if direction == "cost" else f"{tolerance:g}"
        print(f"{name} baseline={prior} current={shown} tolerance={tolerance_text} {status}")
    passed = not failures
    print("gate passed" if passed else "gate failed")
    return 0 if passed else 1


def write_baseline(
    results_path: Path | None = None,
    baseline_path: Path | None = None,
    now: Callable[[], datetime] = _default_now,
) -> dict[str, float]:
    results_path = results_path or gate_results_path()
    baseline_path = baseline_path or ROOT / "eval" / "baseline.json"
    if not results_path.exists():
        raise ConfigError("missing data/gate_results.json")
    results = cast(dict[str, object], json.loads(results_path.read_text(encoding="utf-8")))
    metrics = _baseline_metrics(results)
    value = {
        "written_utc": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": runner._git_commit(),
        "metrics": metrics,
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return metrics
