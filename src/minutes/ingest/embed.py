import psycopg
from psycopg.rows import TupleRow

from minutes.models import get_embedder

BATCH = 64


def run(conn: psycopg.Connection[TupleRow], document_id: str) -> dict[str, object]:
    """Embed the document's chunks that have no vector yet."""
    chunks = conn.execute(
        "SELECT id, embed_text FROM chunks WHERE document_id = %s AND embedding IS NULL "
        "ORDER BY id",
        (document_id,),
    ).fetchall()
    embedder = get_embedder()
    for offset in range(0, len(chunks), BATCH):
        batch = chunks[offset : offset + BATCH]
        vectors = embedder.embed_passages([text for _, text in batch])
        for (chunk_id, _), vector in zip(batch, vectors, strict=True):
            conn.execute("UPDATE chunks SET embedding = %s WHERE id = %s", (vector, chunk_id))
    return {"embedded": len(chunks)}
