import io
import re

import pdfplumber
import psycopg
import pypdfium2
from psycopg.rows import TupleRow

from minutes.errors import NotFoundError


def normalize(text: str) -> str:
    return "\n".join(re.sub(r" +", " ", line).rstrip() for line in text.split("\n"))


def run(conn: psycopg.Connection[TupleRow], document_id: str) -> dict[str, object]:
    """Replace the document's units with one per PDF page; caption files get none yet."""
    row = conn.execute(
        "SELECT file_path, media_type FROM documents WHERE id = %s", (document_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"unknown document {document_id}")
    file_path, media_type = row
    conn.execute("DELETE FROM units WHERE document_id = %s", (document_id,))
    if media_type != "application/pdf":
        return {"units": 0, "needs_ocr": 0}
    with pdfplumber.open(file_path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            conn.execute(
                "INSERT INTO units (document_id, unit_index, unit_kind, text, text_source, "
                "width_pt, height_pt) VALUES (%s, %s, 'page', %s, 'pdf', %s, %s)",
                (document_id, index, normalize(page.extract_text() or ""), page.width, page.height),
            )
        count = len(pdf.pages)
    conn.execute("UPDATE documents SET unit_count = %s WHERE id = %s", (count, document_id))
    return {"units": count, "needs_ocr": 0}


def render_page_png(file_path: str, page_index: int, dpi: int) -> bytes:
    """Render one page (0-based index) to an RGB PNG."""
    document = pypdfium2.PdfDocument(file_path)
    try:
        if not 0 <= page_index < len(document):
            raise NotFoundError(f"page {page_index + 1} not in {file_path}")
        image = document[page_index].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        document.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
