"""Writes the synthetic test corpus from content.json: PDFs, caption files, OCR words, manifest."""

import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pdfplumber
import pypdfium2
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

CORPUS = Path(__file__).parent / "corpus"
WIDTH, HEIGHT = 612, 792
LEFT, RIGHT_COLUMN, TOP, LINE = 72, 330, 720, 14
TABLE_COLUMNS = [72, 200, 330]
MEDIA_TYPES = {"pdf": "application/pdf", "srt": "application/x-subrip", "vtt": "text/vtt"}
# segments the caption parser makes of each transcript: a gap over 5 seconds starts a new one
SEGMENTS = {
    "alder-transcript-101": 2,
    "alder-transcript-102": 1,
    "birch-transcript-201": 2,
    "birch-transcript-202": 1,
}


def new_canvas(target: io.BytesIO) -> Canvas:
    return Canvas(target, pagesize=(WIDTH, HEIGHT), invariant=1, pageCompression=0)


def draw_lines(canvas: Canvas, lines: list[str], x: int) -> None:
    canvas.setFont("Helvetica", 10)
    for i, line in enumerate(lines):
        canvas.drawString(x, TOP - LINE * i, line)


def draw_table(canvas: Canvas, table: dict[str, Any]) -> None:
    rows = table["rows"]
    # each row's box runs from 4 points under its baseline to 10 above it
    top = TOP - LINE * table["after_line"] + 10
    canvas.grid(TABLE_COLUMNS, [top - LINE * i for i in range(len(rows) + 1)])
    for i, row in enumerate(rows):
        for x, cell in zip(TABLE_COLUMNS, row, strict=False):
            canvas.drawString(x + 4, TOP - LINE * (table["after_line"] + i), cell)


def scan(lines: list[str]) -> tuple[ImageReader, list[dict[str, Any]]]:
    """Render the lines as a page image and return it with the words an OCR engine would read."""
    buffer = io.BytesIO()
    canvas = new_canvas(buffer)
    draw_lines(canvas, lines, LEFT)
    canvas.save()
    document = pypdfium2.PdfDocument(buffer.getvalue())
    image = document[0].render(scale=200 / 72, grayscale=True).to_pil()
    document.close()
    with pdfplumber.open(io.BytesIO(buffer.getvalue())) as pdf:
        found = pdf.pages[0].extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
    words = [
        {
            "text": word["text"],
            "confidence": 0.99,
            "x0": round(word["x0"] / WIDTH, 4),
            "y0": round(word["top"] / HEIGHT, 4),
            "x1": round(word["x1"] / WIDTH, 4),
            "y1": round(word["bottom"] / HEIGHT, 4),
        }
        for word in found
    ]
    return ImageReader(image), words


def build_pdf(document: dict[str, Any], out: Path) -> bytes:
    buffer = io.BytesIO()
    canvas = new_canvas(buffer)
    for number, page in enumerate(document["pages"], start=1):
        if page["layout"] == "two_column":
            draw_lines(canvas, page["left"], LEFT)
            draw_lines(canvas, page["right"], RIGHT_COLUMN)
        elif page["layout"] == "scanned":
            image, words = scan(page["lines"])
            canvas.drawImage(image, 0, 0, WIDTH, HEIGHT)
            write_json(out / "ocr" / f"{document['id']}-{number}.json", words)
        else:
            draw_lines(canvas, page["lines"], LEFT)
            if "table" in page:
                draw_table(canvas, page["table"])
        canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def timestamp(ms: int, separator: str) -> str:
    seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def build_srt(cues: list[list[Any]]) -> bytes:
    blocks = [
        f"{i}\n{timestamp(start, ',')} --> {timestamp(end, ',')}\n{text}\n"
        for i, (start, end, text) in enumerate(cues, start=1)
    ]
    return "\n".join(blocks).encode()


def build_vtt(cues: list[list[Any]]) -> bytes:
    blocks = [
        f"{timestamp(start, '.')} --> {timestamp(end, '.')}\n{text}\n" for start, end, text in cues
    ]
    return "\n".join(["WEBVTT\n", *blocks]).encode()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2) + "\n").encode())


def main(out: Path = CORPUS) -> int:
    content = json.loads((CORPUS / "content.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for document in content["documents"]:
        city_id = document["meeting_id"].split("-")[0]
        if document["kind"] != "transcript":
            extension, data, units = "pdf", build_pdf(document, out), len(document["pages"])
        else:
            extension = "srt" if city_id == "alder" else "vtt"
            build = build_srt if extension == "srt" else build_vtt
            data, units = build(document["cues"]), SEGMENTS[document["id"]]
        name = f"{document['id']}.{extension}"
        (out / name).write_bytes(data)
        entries.append(
            {
                "id": document["id"],
                "meeting_id": document["meeting_id"],
                "city_id": city_id,
                "kind": document["kind"],
                "media_type": MEDIA_TYPES[extension],
                "file": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "units": units,
            }
        )
    cities = {
        city_id: {**city, "source": "fixture", "sample_seed": 0}
        for city_id, city in content["cities"].items()
    }
    meetings = [
        {
            **meeting,
            "title": f"{meeting['body']} {meeting['meeting_date']}",
            "source_key": meeting["id"].split("-")[1],
        }
        for meeting in content["meetings"]
    ]
    manifest = {"cities": cities, "meetings": meetings, "documents": entries}
    write_json(out / "manifest.json", manifest)
    pages = sum(e["units"] for e in entries if e["kind"] != "transcript")
    segments = sum(e["units"] for e in entries if e["kind"] == "transcript")
    print(f"wrote {len(entries)} documents, {pages} pages, {segments} segments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
