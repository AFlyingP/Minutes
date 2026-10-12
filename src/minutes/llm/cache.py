import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast


def cache_key(
    model: str, messages: list[dict[str, str]], params: dict[str, object], sample_index: int
) -> str:
    value = {
        "model": model,
        "messages": messages,
        "params": params,
        "sample_index": sample_index,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class DiskCache:
    """JSON cache whose entries are replaced atomically."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            value = json.loads(self._path(key).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        return cast(dict[str, Any], value)

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".llm-cache-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(value, file, ensure_ascii=False, separators=(",", ":"))
                file.write("\n")
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


class ReplayStore:
    """Read recorded response entries from a gzip JSONL file."""

    def __init__(self, path: Path) -> None:
        self.values: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return
        with gzip.open(path, "rt", encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                self.values[row["key"]] = row["value"]

    def get(self, key: str) -> dict[str, Any] | None:
        return self.values.get(key)
