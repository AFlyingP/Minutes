import hashlib
import json
import runpy
from pathlib import Path

import pdfplumber

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"
MANIFEST = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
build = runpy.run_path(str(CORPUS.parent / "build_fixtures.py"))["main"]


def test_build_is_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    assert build(first) == 0
    assert build(second) == 0
    files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    assert len(files) == 14
    for file in files:
        assert (first / file).read_bytes() == (second / file).read_bytes(), file
    captions = [doc for doc in MANIFEST["documents"] if doc["kind"] == "transcript"]
    assert len(captions) == 4
    for doc in captions:
        data = (first / doc["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == doc["sha256"]


def test_manifest_lists_12_documents_and_17_pages() -> None:
    documents = MANIFEST["documents"]
    pdfs = [doc for doc in documents if doc["media_type"] == "application/pdf"]
    assert len(documents) == 12
    assert sum(doc["units"] for doc in pdfs) == 17
    for doc in pdfs:
        with pdfplumber.open(CORPUS / doc["file"]) as pdf:
            assert len(pdf.pages) == doc["units"], doc["id"]


def test_fixture_size_under_limit() -> None:
    total = sum(path.stat().st_size for path in CORPUS.rglob("*") if path.is_file())
    assert total <= 600_000


def test_scanned_page_has_no_text_layer() -> None:
    with pdfplumber.open(CORPUS / "birch-minutes-201.pdf") as pdf:
        page = pdf.pages[2]
        assert (page.extract_text() or "") == ""
        assert len(page.images) == 1
    words = json.loads((CORPUS / "ocr" / "birch-minutes-201-3.json").read_text(encoding="utf-8"))
    assert len(words) > 40
