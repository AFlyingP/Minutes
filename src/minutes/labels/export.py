import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb
from pydantic import ValidationError as PayloadError

from minutes.config import get_settings
from minutes.errors import LabelRuleError, ValidationError
from minutes.labels import store
from minutes.labels.rules import PRODUCTION_RULES, LabelRules, check
from minutes.labels.schema import Label, QuestionAnswerable, parse_payload

LABEL_FILES = {
    "question": "questions.jsonl",
    "agenda_count": "agenda_counts.jsonl",
    "extraction": "extraction.jsonl",
    "agent_task": "agent_tasks.jsonl",
}
Connection = psycopg.Connection[TupleRow]


def document_kinds(conn: Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT id, kind FROM documents").fetchall())


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_labels(path: Path, labels: list[Label]) -> None:
    lines = []
    for label in sorted(labels, key=lambda label: label.id):
        envelope = label.model_dump()
        envelope["created_utc"] = label.created_utc.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        # jsonb does not preserve the payload's field order.
        envelope["payload"] = parse_payload(label.type, label.payload).model_dump()
        lines.append(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def _read_labels(path: Path) -> list[Label]:
    labels = [
        Label.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    for label in labels:
        parse_payload(label.type, label.payload)
    return labels


def _replace(conn: Connection, labels: list[Label]) -> int:
    with conn.transaction():
        conn.execute("DELETE FROM labels")
        for label in labels:
            conn.execute(
                "INSERT INTO labels (id, label_type, city_id, payload, author, created_utc, "
                "human_reviewed, note) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    label.id,
                    label.type,
                    label.city,
                    Jsonb(parse_payload(label.type, label.payload).model_dump()),
                    label.author,
                    label.created_utc,
                    label.human_reviewed,
                    label.note,
                ),
            )
    return len(labels)


def export_type(
    conn: Connection,
    label_type: str,
    labels_dir: Path = Path("eval/labels"),
    rules: LabelRules = PRODUCTION_RULES,
) -> Path:
    store.require_unfrozen(labels_dir)
    if label_type not in LABEL_FILES:
        raise ValidationError(f"unknown label type {label_type}")
    labels = store.list_labels(conn, label_type)
    messages = check(label_type, labels, rules, document_kinds(conn))
    if messages:
        raise LabelRuleError(messages)
    path = labels_dir / LABEL_FILES[label_type]
    _write_labels(path, labels)
    return path


def import_all(conn: Connection, labels_dir: Path = Path("eval/labels")) -> int:
    """Replace the label rows with every tracked label file that exists."""
    labels = []
    for filename in LABEL_FILES.values():
        path = labels_dir / filename
        if path.exists():
            labels.extend(_read_labels(path))
    return _replace(conn, labels)


def backup(conn: Connection) -> Path:
    path = get_settings().data_dir / "labels_backup" / "labels.jsonl"
    _write_labels(path, store.list_labels(conn))
    return path


def restore(conn: Connection) -> int:
    path = get_settings().data_dir / "labels_backup" / "labels.jsonl"
    return _replace(conn, _read_labels(path))


def freeze(labels_dir: Path = Path("eval/labels"), rules: LabelRules = PRODUCTION_RULES) -> None:
    store.require_unfrozen(labels_dir)
    messages = [
        f"missing {name}" for name in LABEL_FILES.values() if not (labels_dir / name).is_file()
    ]
    if messages:
        raise LabelRuleError(messages)
    labels = []
    for label_type, filename in LABEL_FILES.items():
        try:
            cases = _read_labels(labels_dir / filename)
        except PayloadError as err:
            messages.extend(
                f"{filename}: {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in err.errors()
            )
            continue
        if any(label.type != label_type for label in cases):
            messages.append(f"{filename}: contains a different label type")
        labels.extend(cases)
    # Source IDs include their document kind, so the file rules can run without a database.
    doc_kinds = {}
    for label in labels:
        payload = parse_payload(label.type, label.payload)
        if isinstance(payload, QuestionAnswerable):
            for passage in payload.passages:
                doc_kinds[passage.document_id] = passage.document_id.removeprefix(
                    label.city + "-"
                ).split("-", 1)[0]
    for label_type in LABEL_FILES:
        messages.extend(check(label_type, labels, rules, doc_kinds))
    if messages:
        raise LabelRuleError(messages)
    manifest = {
        "frozen": True,
        "frozen_utc": _utc(),
        "files": {
            filename: hashlib.sha256((labels_dir / filename).read_bytes()).hexdigest()
            for filename in LABEL_FILES.values()
        },
        "amendments": [],
    }
    _write_manifest(labels_dir, manifest)


def _write_manifest(labels_dir: Path, manifest: dict[str, object]) -> None:
    (labels_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def amend(file: str, reason: str, labels_dir: Path = Path("eval/labels")) -> None:
    store.require_frozen(labels_dir)
    if file not in LABEL_FILES.values():
        raise ValidationError(f"unknown label file {file}")
    if not 10 <= len(reason) <= 300:
        raise ValidationError("reason must be 10 to 300 characters")
    manifest = json.loads((labels_dir / "manifest.json").read_text(encoding="utf-8"))
    old_sha256 = manifest["files"][file]
    new_sha256 = hashlib.sha256((labels_dir / file).read_bytes()).hexdigest()
    if old_sha256 == new_sha256:
        raise LabelRuleError([f"{file} is unchanged"])
    manifest["files"][file] = new_sha256
    manifest["amendments"].append(
        {
            "utc": _utc(),
            "file": file,
            "old_sha256": old_sha256,
            "new_sha256": new_sha256,
            "reason": reason,
        }
    )
    _write_manifest(labels_dir, manifest)
