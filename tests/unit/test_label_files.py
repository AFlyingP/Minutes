import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from minutes.labels.export import LABEL_FILES
from minutes.labels.rules import PRODUCTION_RULES, QUESTION_TYPES, check
from minutes.labels.schema import Label, QuestionAnswerable, parse_payload

LABELS_DIR = Path(__file__).resolve().parents[2] / "eval" / "labels"


def read_labels(label_type: str) -> list[Label]:
    labels = [
        Label.model_validate_json(line)
        for line in (LABELS_DIR / LABEL_FILES[label_type]).read_text(encoding="utf-8").splitlines()
    ]
    assert all(label.type == label_type for label in labels)
    return labels


def test_label_files_match_frozen_manifest() -> None:
    manifest = json.loads((LABELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen"] is True
    assert set(manifest["files"]) == set(LABEL_FILES.values())
    for filename in LABEL_FILES.values():
        assert (
            hashlib.sha256((LABELS_DIR / filename).read_bytes()).hexdigest()
            == manifest["files"][filename]
        )


def test_question_file_has_60_questions_30_per_city() -> None:
    questions = read_labels("question")
    assert len(questions) == 60
    assert Counter(label.city for label in questions) == {"seattle": 30, "lincoln": 30}
    assert sum(label.payload["answerable"] is False for label in questions) == 12
    for city in ("seattle", "lincoln"):
        cases = [label for label in questions if label.city == city]
        assert sum(label.payload["answerable"] is True for label in cases) == 24
        assert sum(label.payload["answerable"] is False for label in cases) == 6


def test_question_types_are_4_each_per_city() -> None:
    questions = read_labels("question")
    for city in ("seattle", "lincoln"):
        counts = Counter(
            str(label.payload["question_type"])
            for label in questions
            if label.city == city and label.payload["answerable"] is True
        )
        assert counts == {kind: 4 for kind in QUESTION_TYPES}


def test_agenda_count_file_has_20_counts_10_per_city() -> None:
    counts = read_labels("agenda_count")
    assert len(counts) == 20
    assert Counter(label.city for label in counts) == {"seattle": 10, "lincoln": 10}


def test_extraction_file_has_220_cases_110_per_city_with_split() -> None:
    cases = read_labels("extraction")
    assert len(cases) == 220
    assert Counter(label.city for label in cases) == {"seattle": 110, "lincoln": 110}
    for city in ("seattle", "lincoln"):
        assert Counter(str(label.payload["kind"]) for label in cases if label.city == city) == {
            "motion": 20,
            "vote": 30,
            "ordinance": 20,
            "amount": 20,
            "statement": 20,
        }


def test_agent_task_file_has_20_tasks_10_per_city() -> None:
    tasks = read_labels("agent_task")
    assert len(tasks) == 20
    assert Counter(label.city for label in tasks) == {"seattle": 10, "lincoln": 10}


def test_all_production_rules_pass_on_tracked_files() -> None:
    labels = [label for label_type in LABEL_FILES for label in read_labels(label_type)]
    doc_kinds = {}
    for label in labels:
        payload = parse_payload(label.type, label.payload)
        if isinstance(payload, QuestionAnswerable):
            for passage in payload.passages:
                doc_kinds[passage.document_id] = passage.document_id.removeprefix(
                    label.city + "-"
                ).split("-", 1)[0]
    for label_type in LABEL_FILES:
        assert check(label_type, labels, PRODUCTION_RULES, doc_kinds) == []


def test_every_label_has_author_timestamp_review_flag_and_note() -> None:
    for filename in LABEL_FILES.values():
        for line in (LABELS_DIR / filename).read_text(encoding="utf-8").splitlines():
            label = json.loads(line)
            assert label["author"] in {"agent", "human"}
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", label["created_utc"])
            assert isinstance(label["human_reviewed"], bool)
            assert isinstance(label["note"], str)


def test_files_are_sorted_by_id_with_lf_endings() -> None:
    for filename in LABEL_FILES.values():
        content = (LABELS_DIR / filename).read_bytes()
        assert b"\r" not in content
        assert content.endswith(b"\n")
        assert not content.endswith(b"\n\n")
        ids = [json.loads(line)["id"] for line in content.splitlines()]
        assert ids == sorted(ids)
