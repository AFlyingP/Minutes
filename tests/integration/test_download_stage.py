from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from minutes import db, queue, worker
from minutes.config import get_settings, load_cities
from minutes.errors import PermanentStageError
from minutes.ingest import download, pipeline
from minutes.sources import get_source
from minutes.sources.base import DocRef

pytestmark = pytest.mark.integration


@pytest.fixture
def download_db(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[str, Path]]:
    url = (
        urlsplit(get_settings().test_database_url)
        ._replace(path="/minutes_migrations_test")
        .geturl()
    )
    db.reset(url)
    monkeypatch.setenv("MINUTES_TEST_DATABASE_URL", url)
    monkeypatch.setenv("MINUTES_CORPUS", "fixture")
    monkeypatch.setenv("MINUTES_LLM_MODE", "stub")
    monkeypatch.setenv("MINUTES_MODELS_MODE", "stub")
    get_settings.cache_clear()
    data_dir = get_settings().data_dir
    with db.connect(url) as conn:
        pipeline.discover(conn, "fixture", None)
    yield url, data_dir
    get_settings.cache_clear()


def test_download_sets_hash_size_and_path(download_db: tuple[str, Path]) -> None:
    url, _ = download_db
    document_id = "alder-transcript-101"
    with db.connect(url) as conn:
        result = pipeline.run_stage(conn, "download", document_id, "fixture")
        row = conn.execute(
            "SELECT status, content_sha256, byte_size, file_path FROM documents WHERE id = %s",
            (document_id,),
        ).fetchone()

    assert result.status == "done"
    assert row is not None
    status, content_sha256, byte_size, file_path = row
    body = Path(file_path).read_bytes()
    assert status == "downloaded"
    assert len(body) < 2000
    assert content_sha256 == hashlib.sha256(body).hexdigest()
    assert byte_size == len(body)
    assert file_path == "data/raw/alder/alder-transcript-101.srt"


def test_identical_content_marks_duplicate(download_db: tuple[str, Path]) -> None:
    url, data_dir = download_db
    original_id = "alder-agenda-101"
    duplicate_id = "alder-copy-agenda"
    with db.connect(url) as conn:
        original = conn.execute(
            "SELECT meeting_id, city_id, kind, source_url, media_type FROM documents WHERE id = %s",
            (original_id,),
        ).fetchone()
        assert original is not None
        _, city_id, kind, source_url, media_type = original
        pipeline.run_stage(conn, "download", original_id, "fixture")
        conn.execute(
            "INSERT INTO meetings (id, city_id, body, meeting_date, title, source_key) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            ("alder-copy-meeting", city_id, "City Council", "2024-01-01", "copy", "copy-key"),
        )
        conn.execute(
            "INSERT INTO documents (id, meeting_id, city_id, kind, source_url, media_type, "
            "unit_kind) "
            "VALUES (%s, %s, %s, %s, %s, %s, 'page')",
            (duplicate_id, "alder-copy-meeting", city_id, kind, source_url, media_type),
        )
        result = pipeline.run_stage(conn, "download", duplicate_id, "fixture")
        row = conn.execute(
            "SELECT status, duplicate_of, file_path FROM documents WHERE id = %s",
            (duplicate_id,),
        ).fetchone()

    assert result.status == "done"
    assert row == ("duplicate", original_id, None)
    assert not (data_dir / "raw" / "alder" / "alder-copy-agenda.pdf").exists()


def test_rerun_does_not_refetch(
    download_db: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _ = download_db
    real = get_source(
        "alder",
        load_cities("fixture")["alder"],
    )
    calls = 0

    class CountingSource:
        def fetch(self, ref: DocRef) -> bytes:
            nonlocal calls
            calls += 1
            return real.fetch(ref)

    monkeypatch.setattr(download, "get_source", lambda *args: CountingSource())
    with db.connect(url) as conn:
        first = pipeline.run_stage(conn, "download", "alder-agenda-101", "fixture")
        second = pipeline.run_stage(conn, "download", "alder-agenda-101", "fixture")

    assert first.status == "done"
    assert second.status == "skipped"
    assert calls == 1


def test_permanent_error_marks_document_failed(
    download_db: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _ = download_db

    class FailingSource:
        def fetch(self, ref: DocRef) -> bytes:
            raise PermanentStageError("missing fixture")

    monkeypatch.setattr(download, "get_source", lambda *args: FailingSource())
    with db.connect(url) as conn:
        queue.enqueue(conn, "download", "alder-agenda-101", "fixture")

    summary = worker.run("fixture", drain=True, stage="download")
    with db.connect(url) as conn:
        row = conn.execute(
            "SELECT status, fail_reason FROM documents WHERE id = %s", ("alder-agenda-101",)
        ).fetchone()

    assert summary.failed == 1
    assert row == ("failed", "missing fixture")
