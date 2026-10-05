import subprocess
import sys
from pathlib import Path

import minutes

ROOT = Path(__file__).resolve().parents[2]


def test_version_is_set() -> None:
    assert minutes.__version__ == "0.1.0"


def test_tasks_unknown_target_exits_2() -> None:
    result = subprocess.run(
        [sys.executable, "tasks.py", "nope"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "unknown target: nope" in result.stdout


def test_tasks_clean_prints_cleaned(tmp_path: Path) -> None:
    # run in an empty directory so the caches of the running check are left alone
    cache = tmp_path / ".ruff_cache"
    cache.mkdir()
    result = subprocess.run(
        [sys.executable, str(ROOT / "tasks.py"), "clean"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.rstrip().endswith("cleaned")
    assert not cache.exists()
