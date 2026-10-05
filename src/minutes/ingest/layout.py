import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass(frozen=True)
class Table:
    bbox: tuple[float, float, float, float]
    rows: list[list[tuple[float, float, float, float]]]


Box = tuple[int, int, float, float, float, float]


@dataclass(frozen=True)
class _Line:
    top: float
    cells: list[list[Word]]
    table: bool = False


def normalize_text(s: str) -> str:
    """Normalize Unicode, quotes, and horizontal whitespace without joining lines."""
    text = unicodedata.normalize("NFC", s).translate(
        str.maketrans("\u00a0\u2019\u2018\u201c\u201d", " ''\"\"")
    )
    return "\n".join(re.sub(r"[ \t]+", " ", line).rstrip(" ") for line in text.split("\n"))


def _table_lines(words: list[Word], tables: list[Table]) -> tuple[list[Word], list[_Line]]:
    free = list(words)
    lines = []
    for table in tables:
        if len(table.rows) < 2 or max(len(row) for row in table.rows) < 2:
            continue
        cells: list[list[list[Word]]] = [[[] for cell in row] for row in table.rows]
        remaining = []
        for word in free:
            x = (word.x0 + word.x1) / 2
            y = (word.top + word.bottom) / 2
            x0, top, x1, bottom = table.bbox
            if not (x0 <= x < x1 and top <= y < bottom):
                remaining.append(word)
                continue
            for row_index, row in enumerate(table.rows):
                for cell_index, (x0, top, x1, bottom) in enumerate(row):
                    if x0 <= x < x1 and top <= y < bottom:
                        cells[row_index][cell_index].append(word)
                        break
                else:
                    continue
                break
            else:
                remaining.append(word)
        free = remaining
        for row, row_words in zip(table.rows, cells, strict=True):
            for cell_words in row_words:
                cell_words.sort(key=lambda word: (round(word.top), word.x0))
            lines.append(_Line(min(cell[1] for cell in row), row_words, table=True))
    return free, lines


def _lines(words: list[Word]) -> list[_Line]:
    lines: list[_Line] = []
    for word in sorted(words, key=lambda word: (word.top, word.x0)):
        if not lines or abs(word.top - lines[-1].top) > 3:
            lines.append(_Line(word.top, [[word]]))
        else:
            lines[-1].cells[0].append(word)
    for line in lines:
        line.cells[0].sort(key=lambda word: word.x0)
    return lines


def _gutter(words: list[Word], width: float, height: float) -> tuple[float, float] | None:
    body = [word for word in words if 0.08 * height <= word.top <= 0.92 * height]
    bin_width = width / 200
    marked = [
        any(word.x0 < (index + 1) * bin_width and word.x1 > index * bin_width for word in body)
        for index in range(200)
    ]
    gutters = []
    index = 0
    while index < 200:
        if marked[index]:
            index += 1
            continue
        start = index
        while index < 200 and not marked[index]:
            index += 1
        left, right = start * bin_width, index * bin_width
        if (
            start >= 60
            and index <= 140
            and right - left >= 0.025 * width
            and sum(word.x1 <= left for word in body) >= 25
            and sum(word.x0 >= right for word in body) >= 25
        ):
            gutters.append((left, right))
    return min(gutters, key=lambda gutter: (-(gutter[1] - gutter[0]), gutter[0]), default=None)


def _order(lines: list[_Line], gutter: tuple[float, float] | None) -> list[_Line]:
    lines = sorted(lines, key=lambda line: line.top)
    if gutter is None:
        return lines
    left, right = gutter
    center = (left + right) / 2
    ordered: list[_Line] = []
    left_lines: list[_Line] = []
    right_lines: list[_Line] = []
    for line in lines:
        full_width = line.table or any(word.x0 < left and word.x1 > right for word in line.cells[0])
        if full_width:
            ordered.extend(left_lines)
            ordered.extend(right_lines)
            ordered.append(line)
            left_lines.clear()
            right_lines.clear()
            continue
        for is_left, fragments in ((True, left_lines), (False, right_lines)):
            words = [word for word in line.cells[0] if (word.x1 <= center) == is_left]
            if words:
                fragments.append(_Line(line.top, [words]))
    ordered.extend(left_lines)
    ordered.extend(right_lines)
    return ordered


def _emit(lines: list[_Line], width: float, height: float) -> tuple[str, list[Box]]:
    parts = []
    boxes = []
    offset = 0
    for line_index, line in enumerate(lines):
        if line_index:
            parts.append("\n")
            offset += 1
        for cell_index, cell in enumerate(line.cells):
            if cell_index:
                parts.append(" | ")
                offset += 3
            for word_index, word in enumerate(cell):
                if word_index:
                    parts.append(" ")
                    offset += 1
                text = normalize_text(word.text)
                parts.append(text)
                end = offset + len(text)
                boxes.append(
                    (
                        offset,
                        end,
                        round(word.x0 / width, 4),
                        round(word.top / height, 4),
                        round(word.x1 / width, 4),
                        round(word.bottom / height, 4),
                    )
                )
                offset = end
    return "".join(parts), boxes


def build_text(
    words: list[Word], tables: list[Table], page_width: float, page_height: float
) -> tuple[str, list[Box]]:
    """Build reading-order text and word boxes with exact character offsets."""
    free, table_lines = _table_lines(words, tables)
    lines = [*_lines(free), *table_lines]
    gutter = _gutter(free, page_width, page_height)
    return _emit(_order(lines, gutter), page_width, page_height)
