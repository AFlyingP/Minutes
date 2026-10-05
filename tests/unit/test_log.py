import json
import re

import pytest

from minutes.log import configure_logging, get_logger, new_correlation_id


def test_log_line_is_json_with_fixed_keys(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    get_logger("ingest").info("stage done", extra={"event": "stage_end", "fields": {"n": 3}})
    line = json.loads(capsys.readouterr().err)
    assert list(line) == [
        "ts",
        "level",
        "component",
        "event",
        "correlation_id",
        "message",
        "fields",
    ]


def test_correlation_id_is_32_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{32}", new_correlation_id())
