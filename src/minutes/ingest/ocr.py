import io
import os
from threading import Lock
from typing import Any

import numpy as np
import psycopg
from PIL import Image
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb

from minutes.config import get_settings
from minutes.errors import ConfigError
from minutes.ingest import parse
from minutes.ingest.layout import Word, build_text
from minutes.models import OcrWord, get_ocr

_load_lock = Lock()
_predictor: Any = None


def needs_ocr(text: str, image_area_ratio: float) -> bool:
    """Detect image-only pages and text dominated by replacement or private-use characters."""
    count = sum(not char.isspace() for char in text)
    garbled = sum(char == "\ufffd" or "\ue000" <= char <= "\uf8ff" for char in text)
    return (count < 50 and image_area_ratio >= 0.5) or (count > 0 and garbled / count >= 0.2)


class DoctrEngine:  # pragma: no cover
    """Load one shared CUDA predictor when the engine is first requested."""

    def __init__(self) -> None:
        global _predictor
        with _load_lock:
            if _predictor is None:
                os.environ["DOCTR_CACHE_DIR"] = str(get_settings().data_dir / "models" / "doctr")
                try:
                    import torch
                    from doctr.models import ocr_predictor
                except (ImportError, OSError) as err:
                    raise ConfigError(f"gpu models unavailable: {err}") from err
                if not torch.cuda.is_available():
                    raise ConfigError("gpu models unavailable: cuda is unavailable")
                _predictor = (
                    ocr_predictor(
                        det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
                    )
                    .cuda()
                    .eval()
                )
            self._predictor = _predictor

    def read(self, image: Image.Image, hint: str) -> list[OcrWord]:
        result = self._predictor([np.asarray(image.convert("RGB"))])
        return [
            OcrWord(
                word.value,
                float(word.confidence),
                float(word.geometry[0][0]),
                float(word.geometry[0][1]),
                float(word.geometry[1][0]),
                float(word.geometry[1][1]),
            )
            for page in result.pages
            for block in page.blocks
            for line in block.lines
            for word in line.words
        ]


def run(conn: psycopg.Connection[TupleRow], document_id: str) -> dict[str, object]:
    """Replace only flagged page units with OCR text, boxes, and mean confidence."""
    rows = conn.execute(
        "SELECT u.id, u.unit_index, u.width_pt, u.height_pt, d.file_path FROM units u "
        "JOIN documents d ON d.id = u.document_id "
        "WHERE u.document_id = %s AND u.needs_ocr ORDER BY u.unit_index",
        (document_id,),
    ).fetchall()
    for unit_id, unit_index, width, height, file_path in rows:
        png = parse.render_page_png(file_path, unit_index - 1, 200)
        with Image.open(io.BytesIO(png)) as image:
            ocr_words = get_ocr().read(image, f"{document_id}-{unit_index}")
        words = [
            Word(word.text, word.x0 * width, word.y0 * height, word.x1 * width, word.y1 * height)
            for word in ocr_words
        ]
        text, boxes = build_text(words, [], width, height)
        confidence = (
            sum(word.confidence for word in ocr_words) / len(ocr_words) if ocr_words else 0.0
        )
        conn.execute(
            "UPDATE units SET text = %s, boxes = %s, text_source = 'ocr', "
            "ocr_confidence = %s, needs_ocr = false WHERE id = %s",
            (text, Jsonb(boxes), confidence, unit_id),
        )
    return {"ocr_pages": len(rows)}
