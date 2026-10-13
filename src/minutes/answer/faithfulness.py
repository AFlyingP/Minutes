import re
from collections.abc import Callable
from decimal import Decimal

NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _normalized_number(match: str) -> str:
    value = match.replace("$", "").replace(",", "")
    value = value.removesuffix("%")
    return re.sub(r"\.0+$", "", value)


def numeric_tokens(text: str) -> set[str]:
    """Return normalized numeric tokens in text."""
    return {_normalized_number(match[0]) for match in NUMBER.finditer(text)}


def strict_numeric_tokens(text: str) -> set[str]:
    """Return numeric tokens except unmarked small integer counts."""
    tokens = set()
    for found in NUMBER.finditer(text):
        raw = found[0]
        token = _normalized_number(raw)
        if "$" not in raw and "%" not in raw:
            value = Decimal(token)
            if value == value.to_integral_value() and 0 <= value <= 20:
                continue
        tokens.add(token)
    return tokens


def check_answer(
    sentences: list[tuple[str, list[str]]],
    judge: Callable[[list[tuple[str, list[str]]]], list[bool]],
) -> list[bool]:
    """Check source presence and numeric claims before asking the judge."""
    results = [False] * len(sentences)
    eligible_indices = []
    eligible = []
    for index, (sentence, span_texts) in enumerate(sentences):
        if not span_texts:
            continue
        source_numbers = set().union(*(numeric_tokens(text) for text in span_texts))
        if strict_numeric_tokens(sentence) <= source_numbers:
            eligible_indices.append(index)
            eligible.append((sentence, span_texts))
    if eligible:
        for index, verdict in zip(eligible_indices, judge(eligible), strict=False):
            results[index] = verdict
    return results
