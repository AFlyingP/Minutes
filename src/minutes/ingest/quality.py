import re
from dataclasses import dataclass

import psycopg
from psycopg.rows import TupleRow

PUNCTUATION = ".,;:!?()[]\"'$%-\u2013/|*"
ALPHA_TOKEN = re.compile(r"[A-Za-z]+(['\u2019-][A-Za-z]+)*")
NUMBER_TOKEN = re.compile(r"\d+([.,:/-]\d+)*[A-Za-z]{0,2}")
CODE_TOKEN = re.compile(r"[A-Z]{1,5}\d+[A-Za-z0-9-]*")
VOWELS = frozenset("aeiouyAEIOUY")


@dataclass(frozen=True)
class CityQuality:
    city_id: str
    pages: int
    ocr_pages: int
    empty_pages: int
    mean_ocr_confidence: float | None
    parse_quality: float | None


def valid_token_ratio(text: str) -> float:
    """Fraction of stripped whitespace tokens matching the word, number, or code rules."""
    tokens = [token for part in text.split() if (token := part.strip(PUNCTUATION))]
    if not tokens:
        return 0.0
    valid = 0
    for token in tokens:
        if (
            (ALPHA_TOKEN.fullmatch(token) and (len(token) == 1 or any(c in VOWELS for c in token)))
            or NUMBER_TOKEN.fullmatch(token)
            or CODE_TOKEN.fullmatch(token)
        ):
            valid += 1
    return valid / len(tokens)


def parse_quality_report(conn: psycopg.Connection[TupleRow]) -> list[CityQuality]:
    """Summarize page text quality and OCR usage for every city, in city ID order."""
    rows = conn.execute(
        "SELECT c.id, u.text, u.text_source, u.ocr_confidence FROM cities c "
        "LEFT JOIN documents d ON d.city_id = c.id "
        "LEFT JOIN units u ON u.document_id = d.id AND u.unit_kind = 'page' "
        "ORDER BY c.id, d.id, u.unit_index"
    ).fetchall()
    by_city: dict[str, list[tuple[str, str, float | None]]] = {}
    for city_id, text, source, confidence in rows:
        pages = by_city.setdefault(city_id, [])
        if text is not None:
            pages.append((text, source, confidence))
    report = []
    for city_id, pages in by_city.items():
        confidences = [
            confidence
            for _, source, confidence in pages
            if source == "ocr" and confidence is not None
        ]
        report.append(
            CityQuality(
                city_id=city_id,
                pages=len(pages),
                ocr_pages=sum(source == "ocr" for _, source, _ in pages),
                empty_pages=sum(sum(not c.isspace() for c in text) < 50 for text, _, _ in pages),
                mean_ocr_confidence=(sum(confidences) / len(confidences) if confidences else None),
                parse_quality=(
                    sum(valid_token_ratio(text) for text, _, _ in pages) / len(pages)
                    if pages
                    else None
                ),
            )
        )
    return report


def levenshtein(a: str, b: str) -> int:
    """Character edit distance using two rows of dynamic programming."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]
