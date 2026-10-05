"""Task runner. Usage: uv run python tasks.py <target> [flags]"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

WINDOWS = sys.platform == "win32"
MINUTES = ["uv", "run", "minutes"]
API_URL = "http://127.0.0.1:8000"
COVERAGE_FILE = "data/coverage.json"

ENV_STUB = {
    "MINUTES_LLM_MODE": "stub",
    "MINUTES_MODELS_MODE": "stub",
    "MINUTES_CORPUS": "fixture",
    "MINUTES_FAULT": "",
    "MINUTES_OTEL_ENDPOINT": "",
}
ENV_REPLAY = {
    "MINUTES_LLM_MODE": "replay",
    "MINUTES_MODELS_MODE": "replay",
    "MINUTES_CORPUS": "fixture",
}

COVERAGE_THRESHOLDS = [
    ("src/minutes/", 80),
    ("src/minutes/ingest/", 80),
    ("src/minutes/retrieval/", 85),
    ("src/minutes/extract/", 85),
    ("src/minutes/answer/", 85),
    ("src/minutes/labels/", 90),
    ("src/minutes/llm/", 85),
    ("src/minutes/agent/", 80),
    ("src/minutes/evals/", 80),
    ("src/minutes/api/", 80),
]

CLEAN_PATHS = [
    "web/dist",
    "web/coverage",
    "web/test-results",
    "web/playwright-report",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
    "data/coverage.json",
    "data/gate_results.json",
]

CHECK_STAGES: list[tuple[str, set[str]]] = [
    ("fmt", {"--check"}),
    ("lint", set()),
    ("typecheck", set()),
    ("test", set()),
    ("e2e", set()),
    ("gate", set()),
]


def say(line: str) -> None:
    print(line, flush=True)


def child_env(extra: dict[str, str] | None) -> dict[str, str] | None:
    return {**os.environ, **extra} if extra else None


def run(cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None) -> int:
    """Echo and run a command; env holds variables set on top of the current environment."""
    say("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=child_env(env)).returncode


def start(
    cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None
) -> subprocess.Popen[bytes]:
    say("$ " + " ".join(cmd))
    return subprocess.Popen(cmd, cwd=cwd, env=child_env(env))


def stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    if WINDOWS:
        # terminate() would end only the uv or npm launcher and leave its child running
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    proc.wait()


def seq(*steps: Callable[[], int]) -> int:
    for step in steps:
        code = step()
        if code != 0:
            return code
    return 0


def missing(step: str, path: str) -> bool:
    if Path(path).exists():
        return False
    say(f"skip {step}: {path} missing")
    return True


def web(cmd: list[str], env: dict[str, str] | None = None) -> int:
    if missing(" ".join(cmd), "web/package.json"):
        return 0
    tool = cmd[0] + ".cmd" if WINDOWS else cmd[0]
    return run([tool, *cmd[1:]], cwd="web", env={"PLAYWRIGHT_BROWSERS_PATH": "0", **(env or {})})


def compose_up(*services: str) -> int:
    if os.environ.get("MINUTES_SKIP_COMPOSE") == "1":
        return 0
    return run(["docker", "compose", "up", "-d", "--wait", *services])


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, b""
    except OSError:
        return 0, b""


def check_coverage() -> int:
    files = json.loads(Path(COVERAGE_FILE).read_text())["files"]
    code = 0
    for package, threshold in COVERAGE_THRESHOLDS:
        covered = total = 0
        for name, report in files.items():
            if name.replace("\\", "/").startswith(package):
                covered += report["summary"]["covered_lines"]
                total += report["summary"]["num_statements"]
        if total and 100 * covered / total < threshold:
            say(f"coverage {package} {100 * covered / total:.1f} below {threshold}")
            code = 1
    return code


def target_setup(flags: set[str]) -> int:
    sync = ["uv", "sync", "--frozen"]
    if "--gpu" in flags:
        sync += ["--group", "gpu"]
    browsers = ["npx", "playwright", "install", "chromium"]
    if sys.platform == "linux":
        browsers.insert(3, "--with-deps")

    def reset_db() -> int:
        if missing("db reset", "src/minutes/db.py"):
            return 0
        return run([*MINUTES, "db", "reset", "--yes"])

    return seq(
        lambda: run(sync),
        lambda: web(["npm", "ci"]),
        lambda: web(browsers),
        lambda: compose_up("db"),
        reset_db,
    )


def target_fmt(flags: set[str]) -> int:
    if "--check" in flags:
        return seq(
            lambda: run(["uv", "run", "ruff", "format", "--check", "."]),
            lambda: web(["npm", "run", "format:check"]),
        )
    return seq(
        lambda: run(["uv", "run", "ruff", "format", "."]),
        lambda: web(["npm", "run", "format"]),
    )


def target_lint(flags: set[str]) -> int:
    return seq(
        lambda: run(["uv", "run", "ruff", "check", "."]),
        lambda: web(["npm", "run", "lint"]),
    )


def target_typecheck(flags: set[str]) -> int:
    return seq(
        lambda: run(["uv", "run", "mypy"]),
        lambda: web(["npm", "run", "typecheck"]),
    )


def target_test(flags: set[str]) -> int:
    # pytest fails on a path that does not exist, and the integration tests arrive later
    dirs = [d for d in ("tests/unit", "tests/integration") if Path(d).is_dir()]
    pytest = ["uv", "run", "pytest", *dirs, "--cov", "--cov-report=term"]
    pytest.append(f"--cov-report=json:{COVERAGE_FILE}")
    return seq(
        lambda: compose_up("db"),
        lambda: run(pytest, env=ENV_STUB),
        check_coverage,
        lambda: web(["npm", "run", "test"]),
    )


def target_e2e(flags: set[str]) -> int:
    def api() -> int:
        if missing("pytest tests/e2e", "tests/e2e/test_demo_api.py"):
            return 0
        return seq(
            lambda: run([*MINUTES, "fixtures", "load"], env=ENV_STUB),
            lambda: run(["uv", "run", "pytest", "tests/e2e"], env=ENV_STUB),
        )

    return seq(
        api,
        lambda: web(["npm", "run", "build"]),
        lambda: web(["npm", "run", "e2e"], ENV_STUB),
    )


def target_gate(flags: set[str]) -> int:
    if missing("eval gate", "eval/baseline.json"):
        return 0
    started = time.monotonic()
    code = run([*MINUTES, "eval", "gate"], env=ENV_REPLAY)
    if code == 0 and time.monotonic() - started > 180:
        say("gate exceeded 180s")
        return 1
    return code


def target_check(flags: set[str]) -> int:
    durations = {}
    for name, stage_flags in CHECK_STAGES:
        started = time.monotonic()
        code = TARGETS[name](stage_flags)
        durations[name] = time.monotonic() - started
        say(f"stage {name}: {durations[name]:.1f}s")
        if code != 0:
            return code
    total = sum(durations.values())
    say(f"check: {total:.1f}s")
    if durations["test"] > 60:
        say("test exceeded 60s")
        return 1
    if total > 600:
        say("check exceeded 600s")
        return 1
    say("check passed")
    return 0


def target_run(flags: set[str]) -> int:
    code = compose_up("db", "otel")
    if code != 0:
        return code
    api = start([*MINUTES, "serve"])
    npm = "npm.cmd" if WINDOWS else "npm"
    dev = start([npm, "run", "dev"], cwd="web", env={"PLAYWRIGHT_BROWSERS_PATH": "0"})
    say(f"api: {API_URL}")
    say("web: http://127.0.0.1:5173")
    try:
        code = api.wait()
    except KeyboardInterrupt:
        code = 0
    stop(api)
    stop(dev)
    return code


def wait_healthy() -> bool:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if get(f"{API_URL}/api/health")[0] == 200:
            return True
        time.sleep(0.5)
    return False


def smoke() -> int:
    health, _ = get(f"{API_URL}/api/health")
    status, body = get(f"{API_URL}/api/search?q=sidewalk%20repair%20contract&city=birch")
    if health == 200 and status == 200 and json.loads(body)["hits"]:
        say("demo smoke ok")
        return 0
    return 1


def target_demo(flags: set[str]) -> int:
    full = {"--corpus", "full"} <= flags
    code = compose_up("db")
    if code == 0 and not Path("web/dist/index.html").exists():
        code = web(["npm", "run", "build"])
    if code == 0 and not full:
        code = run([*MINUTES, "fixtures", "load"], env=ENV_STUB)
    if code != 0:
        return code
    server = start([*MINUTES, "serve"], env=None if full else ENV_STUB)
    try:
        if not wait_healthy():
            say("demo not ready after 60s")
            return 1
        say(f"demo ready: {API_URL}")
        if "--smoke" in flags:
            return smoke()
        return server.wait()
    except KeyboardInterrupt:
        stop(server)
        say("demo stopped")
        return 0
    finally:
        stop(server)


def target_eval(flags: set[str]) -> int:
    if "--full" not in flags:
        return run([*MINUTES, "eval", "gate"], env=ENV_REPLAY)
    return seq(
        lambda: run([*MINUTES, "eval", "validate-grader"]),
        lambda: run(
            [*MINUTES, "eval", "run", "--suite", "full", "--controls", "off", "--only", "cost"]
        ),
        lambda: run([*MINUTES, "eval", "run", "--suite", "full"]),
        lambda: run([*MINUTES, "eval", "report"]),
    )


def target_clean(flags: set[str]) -> int:
    paths = [Path(p) for p in CLEAN_PATHS]
    for root in ("src", "tests"):
        paths += Path(root).rglob("__pycache__")
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    say("cleaned")
    return 0


def target_reset(flags: set[str]) -> int:
    reset = [*MINUTES, "db", "reset", "--yes"]
    if "--all" in flags:
        reset.append("--all")
    code = seq(lambda: compose_up("db"), lambda: run(reset))
    if code == 0:
        say("reset done")
    return code


TARGETS: dict[str, Callable[[set[str]], int]] = {
    "setup": target_setup,
    "fmt": target_fmt,
    "lint": target_lint,
    "typecheck": target_typecheck,
    "test": target_test,
    "e2e": target_e2e,
    "gate": target_gate,
    "check": target_check,
    "run": target_run,
    "demo": target_demo,
    "eval": target_eval,
    "clean": target_clean,
    "reset": target_reset,
}


def main(argv: list[str]) -> int:
    if not argv:
        say("usage: tasks.py <target> [flags]")
        return 2
    target = TARGETS.get(argv[0])
    if target is None:
        say(f"unknown target: {argv[0]}")
        return 2
    return target(set(argv[1:]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
