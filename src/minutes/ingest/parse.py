import io

import pdfplumber
import psycopg
import pypdfium2
from pdfplumber.page import Page
from pdfplumber.utils.exceptions import PdfminerException
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from minutes.errors import NotFoundError, PermanentStageError
from minutes.ingest import ocr
from minutes.ingest.layout import Table, Word, build_text


def image_area_ratio(page: Page) -> float:
    """Largest image's area as a fraction of the page area."""
    area = max((image["width"] * image["height"] for image in page.images), default=0.0)
    return float(area) / float(page.width * page.height)


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
    try:
        pdf = pdfplumber.open(file_path)
    except PdfminerException as err:
        raise PermanentStageError("unreadable pdf") from err
    ocr_count = 0
    with pdf:
        for index, page in enumerate(pdf.pages, start=1):
            words = [
                Word(word["text"], word["x0"], word["top"], word["x1"], word["bottom"])
                for word in page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
            ]
            tables = [
                Table(
                    table.bbox, [[cell for cell in row.cells if cell is not None] for row in rows]
                )
                for table in page.find_tables()
                if len(rows := table.rows) >= 2 and len(table.columns) >= 2
            ]
            text, boxes = build_text(words, tables, page.width, page.height)
            needs_ocr = ocr.needs_ocr(text, image_area_ratio(page))
            ocr_count += needs_ocr
            conn.execute(
                "INSERT INTO units (document_id, unit_index, unit_kind, text, boxes, text_source, "
                "width_pt, height_pt, needs_ocr) "
                "VALUES (%s, %s, 'page', %s, %s, 'pdf', %s, %s, %s)",
                (document_id, index, text, Jsonb(boxes), page.width, page.height, needs_ocr),
            )
        count = len(pdf.pages)
    conn.execute("UPDATE documents SET unit_count = %s WHERE id = %s", (count, document_id))
    return {"units": count, "needs_ocr": ocr_count}


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
