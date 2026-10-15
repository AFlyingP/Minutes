from typing import cast
from unittest.mock import MagicMock

from minutes.evals import runner


def test_empty_retrieval_reports_null_recall_and_latency() -> None:
    context = runner.EvalContext("full", "full", cast(runner.Connection, MagicMock()), [], True)
    result = runner._retrieval(context)
    fixed = cast(dict[str, dict[str, object]], result["fixed"])
    latency = cast(dict[str, dict[str, object]], result["latency_ms"])
    for mode in ("keyword", "vector"):
        assert fixed[mode] == {"recall_at_5": None, "recall_at_10": None, "recall_at_20": None}
        assert latency[mode] == {"p50": None, "p95": None}
