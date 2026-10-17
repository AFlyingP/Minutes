import re

VOTE_LABELS = ("In Favor:", "Opposed:", "Ayes:", "Noes:", "Nays:", "Abstain:", "Absent:")
MULTI_STEP_PATTERN = re.compile(
    r"\b(compare|comparison|since|between|trend|over time|how many times|how often|"
    r"each|every|all|total|list)\b",
    re.IGNORECASE,
)


def is_easy_question(question: str, top_score: float) -> bool:
    return top_score >= 0.80 and MULTI_STEP_PATTERN.search(question) is None


def is_easy_batch(item_texts: list[str]) -> bool:
    return all(
        len(text) <= 4000 and not any(label.casefold() in text.casefold() for label in VOTE_LABELS)
        for text in item_texts
    )
