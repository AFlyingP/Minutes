import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from minutes import probe
from minutes.cli import app
from minutes.config import get_settings

KEY = "sk-test-0123456789"
SRT = "".join(
    f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},900\nthe clerk called the roll for item {i}\n\n"
    for i in range(1, 50)
)
PAGE = """
playerInstance.setup({
    sources: [{ file: "//video.seattle.gov/media/council/council_010323_2012301V.mp4" }],
    tracks: [{
        file: "documents/seattlechannel/closedcaption/2023/council_010323_2012301.srt",
        label: "English"
    }]
});
"""
EVENTS = [
    {
        "EventId": 5001,
        "EventBodyName": "City Council",
        "EventDate": "2023-01-03T00:00:00",
        "EventAgendaFile": "https://legistar2.granicus.com/seattle/meetings/2023/1/5001_A.pdf",
        "EventMinutesFile": "https://legistar2.granicus.com/seattle/meetings/2023/1/5001_M.pdf",
        "EventMedia": "https://www.seattlechannel.org/FullCouncil?videoid=x144001",
    },
    {
        "EventId": 5002,
        "EventBodyName": "Land Use Committee",
        "EventDate": "2023-01-11T00:00:00",
        "EventAgendaFile": "https://legistar2.granicus.com/seattle/meetings/2023/1/5002_A.pdf",
        "EventMinutesFile": None,
        "EventMedia": None,
    },
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    # an empty directory keeps a developer's .env out of the settings
    for key in os.environ:
        if key.startswith("MINUTES_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(probe, "REQUEST_GAP", 0)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def seattle_only(request: httpx.Request) -> httpx.Response:
    if request.url.host == "webapi.legistar.com":
        return httpx.Response(200, json=EVENTS)
    if request.url.host == "legistar2.granicus.com":
        return httpx.Response(200, content=b"%PDF-1.7 fixture")
    if request.url.host == "www.seattlechannel.org":
        if request.url.path.endswith(".srt"):
            return httpx.Response(200, text=SRT)
        return httpx.Response(200, text=PAGE)
    return httpx.Response(404)


def test_probe_llm_reports_missing_config() -> None:
    results = probe.probe_llm()
    assert len(results) == 1
    assert results[0].ok is False


def test_probe_sources_with_mock_transport() -> None:
    results = {r.name: r for r in probe.probe_sources(httpx.MockTransport(seattle_only))}
    for check in ("listing", "agenda pdf", "minutes pdf", "captions"):
        assert results[f"seattle {check}"].ok, check
    assert results["seattle documents >= 200"].ok is False


def test_probe_never_raises_and_hides_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("MINUTES_LLM_API_KEY", KEY)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused, sent {request.headers['Authorization']}")

    results = probe.probe_llm(httpx.MockTransport(refuse))
    assert len(results) == 4
    for result in results:
        assert result.ok is False
        assert KEY not in result.detail
        assert "***" in result.detail


def test_probe_command_prints_failure_and_exits_1() -> None:
    result = CliRunner().invoke(app, ["probe", "llm"])
    assert result.exit_code == 1
    assert "FAIL config: MINUTES_LLM_BASE_URL and MINUTES_LLM_API_KEY are required" in result.output


def test_probe_command_prints_ok_lines_with_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    passed = [probe.ProbeResult("embedder", True, "dim=768"), probe.ProbeResult("ocr", True, "")]
    monkeypatch.setattr(probe, "probe_gpu", lambda: passed)
    result = CliRunner().invoke(app, ["probe", "gpu"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["ok embedder dim=768", "ok ocr"]


def test_config_error_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINUTES_LLM_MODE", "live")
    result = CliRunner().invoke(app, ["probe", "sources"])
    assert result.exit_code == 2
    assert "error: invalid config: MINUTES_LLM_BASE_URL" in result.output
