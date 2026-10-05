import re

import pytest

from minutes.ingest import pipeline


def test_hash_changes_with_upstream_version(monkeypatch: pytest.MonkeyPatch) -> None:
    content_sha256 = "a" * 64
    download_before = pipeline.stage_input_hash(content_sha256, "download")
    chunk_before = pipeline.stage_input_hash(content_sha256, "chunk")
    monkeypatch.setitem(pipeline.STAGE_VERSIONS, "parse", pipeline.STAGE_VERSIONS["parse"] + 1)

    assert pipeline.stage_input_hash(content_sha256, "download") == download_before
    assert pipeline.stage_input_hash(content_sha256, "chunk") != chunk_before


def test_hash_is_stable() -> None:
    first = pipeline.stage_input_hash("b" * 64, "chunk")
    second = pipeline.stage_input_hash("b" * 64, "chunk")

    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)
