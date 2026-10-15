import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

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


@pytest.fixture
def test_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[ModuleType, Mock]:
    spec = importlib.util.spec_from_file_location("tasks", ROOT / "tasks.py")
    assert spec is not None and spec.loader is not None
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "web").mkdir()
    (tmp_path / "web/package.json").write_text("{}", encoding="utf-8")
    commands = Mock()
    commands.compose_up.return_value = 0
    commands.run.return_value = 0
    commands.check_coverage.return_value = 0
    commands.process.returncode = 0
    commands.process.communicate.return_value = (b"vitest output: \xff\n", None)
    commands.Popen.return_value = commands.process
    monkeypatch.setattr(tasks, "compose_up", commands.compose_up)
    monkeypatch.setattr(tasks, "run", commands.run)
    monkeypatch.setattr(tasks, "check_coverage", commands.check_coverage)
    monkeypatch.setattr(tasks.subprocess, "Popen", commands.Popen)
    return tasks, commands


@pytest.mark.parametrize("windows", [False, True])
def test_test_target_starts_vitest_before_pytest_and_prints_its_output_after(
    test_commands: tuple[ModuleType, Mock], monkeypatch: pytest.MonkeyPatch, windows: bool
) -> None:
    tasks, commands = test_commands
    events: list[str] = []
    process: Mock = commands.process
    monkeypatch.setattr(tasks, "WINDOWS", windows)
    monkeypatch.setattr(tasks, "say", events.append)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "inherited")
    monkeypatch.setenv("MINUTES_TEST_ENV", "preserved")

    def compose_up(*services: str) -> int:
        events.append("db ready")
        return 0

    def popen(*args: object, **kwargs: object) -> Mock:
        events.append("vitest started")
        return process

    def run(*args: object, **kwargs: object) -> int:
        events.append("pytest run")
        return 0

    def coverage() -> int:
        events.append("coverage check")
        return 0

    commands.compose_up.side_effect = compose_up
    commands.Popen.side_effect = popen
    commands.run.side_effect = run
    commands.check_coverage.side_effect = coverage

    assert tasks.target_test(set()) == 0
    npm = "npm.cmd" if windows else "npm"
    assert events == [
        "db ready",
        "vitest started",
        "pytest run",
        "coverage check",
        f"$ {npm} run test",
        "vitest output: \ufffd\n",
    ]
    commands.compose_up.assert_called_once_with("db")
    commands.Popen.assert_called_once_with(
        [npm, "run", "test"],
        cwd="web",
        env=tasks.child_env({"PLAYWRIGHT_BROWSERS_PATH": "0"}),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    env = commands.Popen.call_args.kwargs["env"]
    assert env["PLAYWRIGHT_BROWSERS_PATH"] == "0"
    assert env["MINUTES_TEST_ENV"] == "preserved"
    commands.run.assert_called_once_with(
        [
            "uv",
            "run",
            "pytest",
            "--cov",
            "--cov-report=term",
            "--cov-report=json:data/coverage.json",
        ],
        env=tasks.ENV_STUB,
    )
    commands.process.communicate.assert_called_once_with()


@pytest.mark.parametrize("coverage_code", [0, 3])
def test_test_target_fails_when_vitest_fails(
    test_commands: tuple[ModuleType, Mock], coverage_code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks, commands = test_commands
    commands.check_coverage.return_value = coverage_code
    commands.process.returncode = 1

    assert tasks.target_test(set()) == (coverage_code or 1)
    commands.check_coverage.assert_called_once_with()
    commands.process.communicate.assert_called_once_with()
    assert "vitest output: \ufffd" in capsys.readouterr().out


def test_test_target_waits_for_vitest_when_pytest_fails(
    test_commands: tuple[ModuleType, Mock], capsys: pytest.CaptureFixture[str]
) -> None:
    tasks, commands = test_commands
    commands.run.return_value = 1
    commands.process.returncode = 2

    assert tasks.target_test(set()) == 1
    commands.check_coverage.assert_not_called()
    commands.process.communicate.assert_called_once_with()
    assert "vitest output: \ufffd" in capsys.readouterr().out


def test_test_target_without_web_runs_no_vitest(
    test_commands: tuple[ModuleType, Mock],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tasks, commands = test_commands
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    assert tasks.target_test(set()) == 0
    commands.Popen.assert_not_called()
    commands.run.assert_called_once()
    commands.check_coverage.assert_called_once_with()
    assert capsys.readouterr().out == "skip npm run test: web/package.json missing\n"
