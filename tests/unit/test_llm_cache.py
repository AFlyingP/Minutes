import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from minutes import llm
from minutes.config import get_settings
from minutes.llm.cache import DiskCache, ReplayStore, cache_key
from minutes.llm.client import ThinClient


def test_key_is_stable_and_order_independent() -> None:
    messages = [{"role": "user", "content": "hello"}]
    first = cache_key(
        "gpt-6-luna",
        messages,
        {"reasoning_effort": "low", "max_completion_tokens": 200},
        3,
    )
    second = cache_key(
        "gpt-6-luna",
        messages,
        {"max_completion_tokens": 200, "reasoning_effort": "low"},
        3,
    )
    assert first == second
    assert first == cache_key(
        "gpt-6-luna",
        messages,
        {"max_completion_tokens": 200, "reasoning_effort": "low"},
        3,
    )


def test_disk_cache_round_trip_and_path_layout(tmp_path: Path) -> None:
    key = "ab" + "c" * 62
    value = {
        "text": "reply",
        "model": "gpt-6-luna",
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "latency_ms": 12,
    }
    cache = DiskCache(tmp_path / "cache")
    assert cache.get(key) is None
    cache.put(key, value)
    assert cache.get(key) == value
    assert (tmp_path / "cache" / "ab" / f"{key}.json").is_file()


def test_replay_store_reads_gzip_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl.gz"
    row = {
        "key": "digest",
        "value": {
            "text": "saved",
            "model": "gpt-6.1-sol",
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "latency_ms": 7,
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")
    store = ReplayStore(path)
    assert store.get("digest") == row["value"]
    assert store.get("missing") is None


def test_replay_clients_reuse_recording_and_notice_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MINUTES_LLM_MODE", "replay")
    monkeypatch.setattr(llm, "REPLAY_DIR", tmp_path)
    get_settings.cache_clear()
    path = tmp_path / "llm.jsonl.gz"

    def write(text: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as file:
            file.write(json.dumps({"key": "digest", "value": {"text": text}}) + "\n")

    try:
        write("first")
        with patch("gzip.open", wraps=gzip.open) as read, patch("httpx.Client"):
            first = llm.get_client()
            second = llm.get_client()
            assert isinstance(first, ThinClient)
            assert isinstance(second, ThinClient)
            assert isinstance(first.cache, ReplayStore)
            assert isinstance(second.cache, ReplayStore)
            assert first.cache.values is second.cache.values
            assert read.call_count == 1
        stamp = path.stat().st_mtime_ns
        write("other")
        os.utime(path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
        with patch("httpx.Client"):
            changed = llm.get_client()
        assert isinstance(changed, ThinClient)
        assert isinstance(changed.cache, ReplayStore)
        assert changed.cache.get("digest") == {"text": "other"}
        assert changed.cache.values is not first.cache.values
        stamp = path.stat().st_mtime_ns
        size = path.stat().st_size
        write("replacement with a different length")
        assert path.stat().st_size != size
        os.utime(path, ns=(stamp, stamp))
        with patch("httpx.Client"):
            resized = llm.get_client()
        assert isinstance(resized, ThinClient)
        assert isinstance(resized.cache, ReplayStore)
        assert resized.cache.get("digest") == {"text": "replacement with a different length"}
    finally:
        get_settings.cache_clear()
