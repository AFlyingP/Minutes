import io

import numpy as np
import psycopg
import pytest
from PIL import Image
from psycopg.rows import TupleRow
from typer.testing import CliRunner

from minutes.cli import app
from minutes.config import get_settings
from minutes.errors import ConfigError, NotFoundError
from minutes.ingest import pipeline
from minutes.ingest.parse import render_page_png
from minutes.models import get_embedder, get_ocr, get_reranker

pytestmark = pytest.mark.integration

Connection = psycopg.Connection[TupleRow]
COUNTS = (
    "SELECT (SELECT count(*) FROM documents WHERE status = 'downloaded'), "
    "(SELECT count(*) FROM units), (SELECT count(*) FROM items), (SELECT count(*) FROM chunks)"
)


def test_fixture_load_counts(fixture_corpus: str, conn: Connection) -> None:
    counts = conn.execute(COUNTS).fetchone()
    embeddings = conn.execute(
        "SELECT count(*) FILTER (WHERE embedding IS NULL), min(vector_dims(embedding)) FROM chunks"
    ).fetchone()
    assert counts is not None
    assert counts[:3] == (12, 23, 8)
    assert counts[3] > 0
    assert embeddings == (0, 768)


def test_second_run_skips_every_stage(fixture_corpus: str, conn: Connection) -> None:
    before = conn.execute(COUNTS).fetchone()
    for document_id in pipeline.documents_for(conn, "fixture", None):
        for stage in pipeline.STAGES:
            assert pipeline.run_stage(conn, stage, document_id, "fixture").status == "skipped"
    assert conn.execute(COUNTS).fetchone() == before


def test_stage_version_bump_reruns_downstream(
    fixture_corpus: str, conn: Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(pipeline.STAGE_VERSIONS, "chunk", 2)
    statuses = {
        stage: pipeline.run_stage(conn, stage, "alder-minutes-101", "fixture").status
        for stage in pipeline.STAGES
    }
    assert statuses["parse"] == "skipped"
    assert statuses["segment"] == "skipped"
    assert statuses["chunk"] == "done"
    assert statuses["embed"] == "done"


def test_stub_embedder_is_deterministic() -> None:
    embedder = get_embedder()
    first = embedder.embed_query("sidewalk repair contract")
    second = embedder.embed_passages(["sidewalk repair contract"])[0]
    assert np.array_equal(first, second)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-6
    assert embedder.embed_query("the of")[0] == 1.0


def test_stub_reranker_scores_shared_query_tokens() -> None:
    scores = get_reranker().score("sidewalk repair", ["Oak Street Sidewalk project", "zoning"])
    assert scores == [0.5, 0.0]
    assert get_reranker().score("the", ["anything"]) == [0.0]


def test_stub_ocr_reads_the_sidecar_words(fixture_corpus: str) -> None:
    page = render_page_png("tests/fixtures/corpus/birch-minutes-201.pdf", 2, 72)
    assert page.startswith(b"\x89PNG")
    image = Image.open(io.BytesIO(page))
    ocr = get_ocr()
    assert ocr.read(image, "birch-minutes-201-3")[0].text == "Page"
    assert ocr.read(image, "birch-minutes-201-1") == []


def test_render_page_png_rejects_a_page_past_the_end() -> None:
    with pytest.raises(NotFoundError):
        render_page_png("tests/fixtures/corpus/birch-minutes-201.pdf", 3, 72)


def test_unknown_document_is_not_found(fixture_corpus: str, conn: Connection) -> None:
    with pytest.raises(NotFoundError):
        pipeline.run_stage(conn, "parse", "alder-minutes-999", "fixture")


def test_gpu_embedder_is_not_available_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_MODELS_MODE", "gpu")
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigError, match="models mode gpu is not available"):
            get_embedder()
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_discover_command_finds_nothing_new(fixture_corpus: str) -> None:
    result = CliRunner().invoke(app, ["ingest", "discover", "--city", "birch"])
    assert result.output == "discovered 0 new documents\n"


def test_fixtures_load_command_reloads_the_corpus(fixture_corpus: str) -> None:
    result = CliRunner().invoke(app, ["fixtures", "load"])
    assert result.output == "loaded 12 documents\n"
