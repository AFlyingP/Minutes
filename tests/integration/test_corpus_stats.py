import re

import pytest
from typer.testing import CliRunner

from minutes.cli import app
from minutes.corpus import CityStats, Connection, CorpusStats, assert_minimums, corpus_stats
from minutes.errors import CorpusError

pytestmark = pytest.mark.integration


def test_fixture_stats_counts(fixture_corpus: str, conn: Connection) -> None:
    stats = corpus_stats(conn)
    assert list(stats.per_city) == ["alder", "birch"]
    alder = stats.per_city["alder"]
    assert alder.documents == 6
    assert alder.agenda == alder.minutes == alder.transcript == 2
    assert alder.pages == 8
    assert alder.segments == 3
    assert alder.ocr_pages == 0
    birch = stats.per_city["birch"]
    assert birch.documents == 6
    assert birch.agenda == birch.minutes == birch.transcript == 2
    assert birch.pages == 9
    assert birch.segments == 3
    assert birch.ocr_pages == 1
    assert stats.total_pages == 17
    byte_sizes = conn.execute(
        "SELECT city_id, byte_size FROM documents WHERE status = 'downloaded'"
    ).fetchall()
    for city_id, city in stats.per_city.items():
        assert city.duplicate == city.skipped == city.failed == 0
        assert city.bytes == sum(
            size for document_city, size in byte_sizes if document_city == city_id
        )
        assert city.bytes > 0


def test_assert_minimums_fails_on_fixture_with_message(
    fixture_corpus: str, conn: Connection
) -> None:
    with pytest.raises(CorpusError, match=r"^alder has 6 documents, minimum is 200$"):
        assert_minimums(corpus_stats(conn))
    result = CliRunner().invoke(
        app, ["corpus", "stats", "--corpus", "fixture", "--assert-minimums"]
    )
    assert result.exit_code == 1
    assert result.stderr == "error: alder has 6 documents, minimum is 200\n"
    assert result.stdout.splitlines()[-1] == "total pages=17"


def test_assert_minimums_passes_for_large_counts() -> None:
    stats = CorpusStats(
        per_city={
            "alder": CityStats(documents=200, pages=1500),
            "birch": CityStats(documents=250, pages=1500),
        },
        total_pages=3000,
    )
    assert_minimums(stats)


def test_stats_cli_line_format(fixture_corpus: str) -> None:
    result = CliRunner().invoke(app, ["corpus", "stats", "--corpus", "fixture"])
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert len(lines) == 3
    assert re.fullmatch(
        r"alder documents=6 agenda=2 minutes=2 transcript=2 "
        r"pages=8 segments=3 ocr_pages=0 bytes=\d+",
        lines[0],
    )
    assert re.fullmatch(
        r"birch documents=6 agenda=2 minutes=2 transcript=2 "
        r"pages=9 segments=3 ocr_pages=1 bytes=\d+",
        lines[1],
    )
    assert lines[-1] == "total pages=17"
