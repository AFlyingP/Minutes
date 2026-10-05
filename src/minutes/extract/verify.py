import re
from dataclasses import dataclass
from typing import Any

CATEGORY_LABELS = {
    "ayes": "In Favor|Ayes|Aye|Yeas|Yes",
    "noes": "Opposed|Noes|Nays|No",
    "abstain": "Abstain|Abstained|Abstentions|Abstaining",
    "absent": "Absent|Excused",
}
CATEGORY_WORDS = {
    "ayes": "Aye|Yes|Yea",
    "noes": "No|Nay",
    "abstain": "Abstain|Abstained",
    "absent": "Absent|Excused",
}
MEMBER_VALUES = {"ayes": "aye", "noes": "no", "abstain": "abstain", "absent": "absent"}
LABEL = re.compile(
    r"\b(?:" + "|".join(f"(?P<{c}>{labels})" for c, labels in CATEGORY_LABELS.items()) + r"):"
)
TABLE_LINE = re.compile(
    r"\|\s*(?:" + "|".join(f"(?P<{c}>{words})" for c, words in CATEGORY_WORDS.items()) + r")\s*$",
    re.I,
)
AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(million|billion|thousand)?")


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    error: str | None
    quote_start: int | None
    quote_end: int | None


def ws(s: str) -> str:
    return " ".join(s.split())


def labelled(text: str) -> list[tuple[str, str]]:
    matches = list(LABEL.finditer(text))
    starts = [m.start() for m in matches[1:]]
    return [
        (str(m.lastgroup), text[m.end() : end])
        for m, end in zip(matches, [*starts, len(text)], strict=False)
    ]


def names(listed: str) -> list[str]:
    listed = re.sub(r"^\s*\d{1,2}\s*[-\u2013]", "", listed).strip()
    if listed.startswith("None"):
        return []
    return [name.strip() for name in listed.split(",") if name.strip()]


def quoted_counts(quote: str) -> dict[str, int]:
    """Count votes from label lists and table rows in a quote."""
    counts = dict.fromkeys(CATEGORY_LABELS, 0)
    for category, listed in labelled(ws(quote)):
        number = re.search(r"\b(\d{1,2})\s*[-\u2013]", listed)
        counts[category] += int(number[1]) if number else len(names(listed))
    for line in quote.split("\n"):
        row = TABLE_LINE.search(ws(line))
        if row:
            counts[str(row.lastgroup)] += 1
    return counts


def quoted_amounts(quote: str) -> list[float]:
    multipliers = {None: 1, "million": 1e6, "billion": 1e9, "thousand": 1e3}
    return [
        float(match[1].replace(",", "")) * multipliers[match[2]]
        for match in AMOUNT.finditer(ws(quote))
    ]


def verify(kind: str, data: dict[str, Any], item_text: str) -> VerifyResult:
    """Verify a fact against its quote and map the quote back to source offsets."""
    if kind not in ("motion", "vote", "ordinance", "amount", "statement"):
        raise ValueError(f"unknown fact kind {kind}")
    quote = ws(data["quote"])
    tokens = list(re.finditer(r"\S+", item_text))
    normalized = " ".join(token[0] for token in tokens)
    start = normalized.find(quote)
    if start < 0 or not quote:
        return VerifyResult(False, "quote_not_found", None, None)
    # Spaces in the normalized text cover whole whitespace runs in the source.
    offsets: list[int] = []
    for token in tokens:
        if offsets:
            offsets.append(token.start() - 1)
        offsets.extend(range(token.start(), token.end()))
    quote_start = offsets[start]
    quote_end = offsets[start + len(quote) - 1] + 1
    error = None
    if kind == "vote":
        if any(data[category] != count for category, count in quoted_counts(data["quote"]).items()):
            error = "vote_count_mismatch"
        elif data["members"] and any(
            sum(member["value"] == value for member in data["members"]) != data[category]
            for category, value in MEMBER_VALUES.items()
        ):
            error = "vote_members_mismatch"
    elif kind == "amount":
        if not any(abs(amount - data["amount_usd"]) <= 0.005 for amount in quoted_amounts(quote)):
            error = "amount_not_in_quote"
    elif kind == "ordinance":
        if ws(data["identifier"]).upper() not in quote.upper():
            error = "identifier_not_in_quote"
    else:
        names = [data["speaker"]] if kind == "statement" else [data["mover"], data["seconder"]]
        if any(
            name is not None
            and (not name.split() or name.split()[-1].casefold() not in quote.casefold())
            for name in names
        ):
            error = "name_not_in_quote"
    return VerifyResult(error is None, error, quote_start, quote_end)
