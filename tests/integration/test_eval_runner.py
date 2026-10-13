import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from minutes.errors import ValidationError
from minutes.evals.report import render_table
from minutes.evals.runner import run_suite

pytestmark = pytest.mark.integration
LABELS_DIR = Path("tests/fixtures/labels")


@dataclass(frozen=True)
class EvalFile:
    path: Path
    results: dict[str, object]


@pytest.fixture(scope="module")
def eval_file(fixture_corpus: str, tmp_path_factory: pytest.TempPathFactory) -> EvalFile:
    output = tmp_path_factory.mktemp("eval") / "full.json"
    run_suite(
        suite="full",
        controls=True,
        labels_dir=LABELS_DIR,
        out_path=output,
    )
    return EvalFile(output, cast(dict[str, object], json.loads(output.read_text(encoding="utf-8"))))


def test_runner_writes_file_with_required_keys(eval_file: EvalFile) -> None:
    assert eval_file.path.is_file()
    assert {
        "suite",
        "generated_utc",
        "git_commit",
        "corpus",
        "labels",
        "retrieval",
        "segmentation",
    } <= eval_file.results.keys()
    segmentation = cast(dict[str, object], eval_file.results["segmentation"])
    assert "per_city" in segmentation
    assert "per_meeting" in segmentation


def test_labels_block_counts_fixture_labels(eval_file: EvalFile) -> None:
    labels = cast(dict[str, object], eval_file.results["labels"])
    assert labels["question"] == 4
    assert labels["agenda_count"] == 4
    assert labels["extraction"] == 6
    assert labels["agent_task"] == 2


def test_segmentation_before_real_segmenter(eval_file: EvalFile) -> None:
    segmentation = cast(dict[str, object], eval_file.results["segmentation"])
    overall = cast(dict[str, object], segmentation["overall"])
    assert overall["exact_rate"] == 0.0
    assert overall["mae"] == 3.25


def test_retrieval_block_has_three_k_values_per_mode(eval_file: EvalFile) -> None:
    retrieval = cast(dict[str, object], eval_file.results["retrieval"])
    fixed = cast(dict[str, object], retrieval["fixed"])
    for mode in ("keyword", "vector"):
        values = cast(dict[str, float], fixed[mode])
        assert {"recall_at_5", "recall_at_10", "recall_at_20"} <= values.keys()
        assert 0.0 <= values["recall_at_5"] <= values["recall_at_10"]
        assert values["recall_at_10"] <= values["recall_at_20"] <= 1.0


def test_only_updates_named_section_and_keeps_others(
    eval_file: EvalFile, tmp_path: Path, fixture_corpus: str
) -> None:
    output = tmp_path / "existing.json"
    output.write_text(json.dumps(eval_file.results), encoding="utf-8")
    run_suite(
        suite="full",
        controls=True,
        labels_dir=LABELS_DIR,
        out_path=output,
        only={"segmentation"},
    )
    merged = cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))
    assert merged["retrieval"] == eval_file.results["retrieval"]
    assert merged["segmentation"] == eval_file.results["segmentation"]


def test_two_runs_differ_only_in_timestamp_and_commit(tmp_path: Path, fixture_corpus: str) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    for output in (first_path, second_path):
        run_suite(
            suite="full",
            controls=True,
            labels_dir=LABELS_DIR,
            out_path=output,
        )
    first = cast(dict[str, object], json.loads(first_path.read_text(encoding="utf-8")))
    second = cast(dict[str, object], json.loads(second_path.read_text(encoding="utf-8")))
    first.pop("generated_utc")
    second.pop("generated_utc")
    first.pop("git_commit")
    second.pop("git_commit")
    assert first == second


def test_unknown_section_is_rejected(tmp_path: Path, fixture_corpus: str) -> None:
    with pytest.raises(ValidationError, match=r"^unknown section extraction$"):
        run_suite(
            suite="full",
            controls=True,
            labels_dir=LABELS_DIR,
            out_path=tmp_path / "unknown.json",
            only={"extraction"},
        )


def _numbers(value: object) -> set[float]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        return {round(float(value), 4)}
    if isinstance(value, dict):
        return set().union(*(_numbers(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_numbers(item) for item in value))
    return set()


def test_render_table_numbers_come_from_results(eval_file: EvalFile) -> None:
    table = render_table(eval_file.results)
    result_numbers = _numbers(eval_file.results)
    for line in table.splitlines()[2:]:
        cells = line.strip("|").split("|")
        for cell in cells[1:3]:
            for number in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", cell):
                assert round(float(number), 4) in result_numbers
    without_sections = render_table({"suite": "full"})
    assert "| recall@10 keyword | n/a | n/a | full |" in without_sections
    assert "| answer accuracy final | n/a | n/a | full |" in without_sections
