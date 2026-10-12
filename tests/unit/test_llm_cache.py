import gzip
import json
from pathlib import Path

from minutes.llm.cache import DiskCache, ReplayStore, cache_key


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
