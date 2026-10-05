import hashlib

import psycopg
from psycopg.rows import TupleRow

from minutes.config import get_settings, load_cities
from minutes.errors import NotFoundError
from minutes.sources import get_source
from minutes.sources.base import DocRef

Connection = psycopg.Connection[TupleRow]
EXTENSIONS = {"application/pdf": "pdf", "application/x-subrip": "srt", "text/vtt": "vtt"}


def run(conn: Connection, document_id: str) -> dict[str, object]:
    row = conn.execute(
        "SELECT meeting_id, city_id, kind, source_url, media_type FROM documents WHERE id = %s",
        (document_id,),
    ).fetchone()
    if row is None:
        raise NotFoundError(f"unknown document {document_id}")
    meeting_id, city_id, kind, source_url, media_type = row
    city = load_cities()[city_id]
    source = get_source(city_id, city)
    data = source.fetch(DocRef(document_id, meeting_id, city_id, kind, source_url, media_type))

    digest = hashlib.sha256(data).hexdigest()
    if media_type != "application/pdf" and len(data) < 2000 and city.source != "fixture":
        conn.execute(
            "UPDATE documents SET status = 'skipped', duplicate_of = NULL, content_sha256 = NULL, "
            "byte_size = NULL, file_path = NULL, fail_reason = 'empty_captions', "
            "updated_at = now() "
            "WHERE id = %s",
            (document_id,),
        )
        return {"status": "skipped", "bytes": len(data)}

    original = conn.execute(
        "SELECT id FROM documents WHERE city_id = %s AND content_sha256 = %s "
        "AND status = 'downloaded' AND id <> %s",
        (city_id, digest, document_id),
    ).fetchone()
    if original is not None:
        conn.execute(
            "UPDATE documents SET status = 'duplicate', duplicate_of = %s, content_sha256 = %s, "
            "byte_size = %s, file_path = NULL, fail_reason = NULL, "
            "updated_at = now() WHERE id = %s",
            (original[0], digest, len(data), document_id),
        )
        return {"status": "duplicate", "bytes": len(data)}

    path = get_settings().data_dir / "raw" / city_id / f"{document_id}.{EXTENSIONS[media_type]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        path.write_bytes(data)
    conn.execute(
        "UPDATE documents SET status = 'downloaded', duplicate_of = NULL, content_sha256 = %s, "
        "byte_size = %s, file_path = %s, fail_reason = NULL, updated_at = now() WHERE id = %s",
        (digest, len(data), path.as_posix(), document_id),
    )
    return {"status": "downloaded", "bytes": len(data)}
