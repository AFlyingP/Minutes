from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, Response

from minutes.api.routes_core import Conn
from minutes.api.schemas import (
    LabelExportIn,
    LabelExportOut,
    LabelProgress,
    LabelsOut,
    LabelsProgressOut,
    ReviewIn,
    SampleMeetingOut,
    SampleMeetingsOut,
    TextMatchOut,
    TextSearchOut,
)
from minutes.errors import ValidationError
from minutes.labels import export, store
from minutes.labels.rules import PRODUCTION_RULES, check
from minutes.labels.schema import Label, LabelIn, LabelType

router = APIRouter(prefix="/api/labels")
LABELS_DIR = Path("eval/labels")
RULES = PRODUCTION_RULES


def _check_city(conn: Conn, city: str | None) -> None:
    if (
        city is not None
        and not conn.execute("SELECT 1 FROM cities WHERE id = %s", (city,)).fetchone()
    ):
        raise ValidationError(f"city: unknown city {city}")


@router.get("")
def list_labels(conn: Conn, type: LabelType | None = None, city: str | None = None) -> LabelsOut:
    _check_city(conn, city)
    return LabelsOut(labels=store.list_labels(conn, type, city))


@router.post("", status_code=201)
def create_label(conn: Conn, request: LabelIn) -> Label:
    return store.create(conn, request, labels_dir=LABELS_DIR)


@router.get("/progress")
def progress(conn: Conn) -> LabelsProgressOut:
    labels = store.list_labels(conn)
    doc_kinds = export.document_kinds(conn)
    return LabelsProgressOut(
        frozen=store.is_frozen(LABELS_DIR),
        types={
            label_type: LabelProgress(
                count=sum(label.type == label_type for label in labels),
                violations=check(label_type, labels, RULES, doc_kinds),
            )
            for label_type in export.LABEL_FILES
        },
        human_reviewed={
            label_type: sum(label.type == label_type and label.human_reviewed for label in labels)
            for label_type in export.LABEL_FILES
        },
    )


@router.get("/text-search")
def text_search(
    conn: Conn, city: str, q: Annotated[str, Query(min_length=2, max_length=100)]
) -> TextSearchOut:
    _check_city(conn, city)
    result = store.text_search(conn, city, q)
    return TextSearchOut(
        unit_count=result.unit_count,
        matches=[
            TextMatchOut(document_id=document, unit_index=index, snippet=snippet)
            for document, index, snippet in result.matches
        ],
    )


@router.get("/sample-meetings")
def sample_meetings(conn: Conn, city: str) -> SampleMeetingsOut:
    _check_city(conn, city)
    ids = store.sample_meetings(conn, city)
    rows = conn.execute(
        "SELECT m.id, m.body, m.meeting_date, d.id, EXISTS ("
        "SELECT 1 FROM labels l WHERE l.label_type = 'agenda_count' "
        "AND l.payload->>'meeting_id' = m.id) "
        "FROM meetings m JOIN documents d ON d.meeting_id = m.id "
        "WHERE m.id = ANY(%s) AND d.kind = 'agenda' AND d.status = 'downloaded'",
        (ids,),
    ).fetchall()
    meetings = {
        row[0]: SampleMeetingOut(
            meeting_id=row[0],
            body=row[1],
            meeting_date=row[2],
            agenda_document_id=row[3],
            labelled=row[4],
        )
        for row in rows
    }
    return SampleMeetingsOut(meetings=[meetings[meeting] for meeting in ids])


@router.post("/export")
def export_labels(conn: Conn, request: LabelExportIn) -> LabelExportOut:
    path = export.export_type(conn, request.type, LABELS_DIR, RULES)
    return LabelExportOut(path=path.as_posix(), count=len(store.list_labels(conn, request.type)))


@router.get("/{id}")
def get_label(conn: Conn, id: str) -> Label:
    return store.get(conn, id)


@router.put("/{id}")
def update_label(conn: Conn, id: str, request: LabelIn) -> Label:
    return store.update(conn, id, request, labels_dir=LABELS_DIR)


@router.delete("/{id}", status_code=204)
def delete_label(conn: Conn, id: str) -> Response:
    store.delete(conn, id, labels_dir=LABELS_DIR)
    return Response(status_code=204)


@router.post("/{id}/review")
def review_label(conn: Conn, id: str, request: ReviewIn) -> Label:
    return store.set_reviewed(conn, id, request.human_reviewed, labels_dir=LABELS_DIR)
