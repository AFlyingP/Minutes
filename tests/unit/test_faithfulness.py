from minutes.answer.faithfulness import check_answer, numeric_tokens, strict_numeric_tokens


def test_numeric_tokens_normalize_currency_commas_percent() -> None:
    assert numeric_tokens("$184,500.00 and 184500, then 12%") == {"184500", "12"}


def test_strict_tokens_drop_small_bare_integers_only() -> None:
    assert strict_numeric_tokens("4 to 1") == set()
    assert strict_numeric_tokens("$4 and 25") == {"4", "25"}


def test_sentence_with_number_missing_from_sources_is_unsupported_without_judge_call() -> None:
    calls = []

    def judge(sentences: list[tuple[str, list[str]]]) -> list[bool]:
        calls.append(sentences)
        return [True]

    result = check_answer(
        [("The cost was $25.", ["The contract cost $20."])],
        judge,
    )

    assert result == [False]
    assert calls == []


def test_sentence_without_citations_is_unsupported() -> None:
    assert check_answer([("The vote passed.", [])], lambda _: [True]) == [False]


def test_judge_called_once_for_remaining_sentences_in_order() -> None:
    seen = []
    sentences = [
        ("The vote was 4 to 1.", ["The vote was 4 to 1."]),
        ("The amount was $20.", ["The amount was $20."]),
    ]

    def judge(eligible: list[tuple[str, list[str]]]) -> list[bool]:
        seen.append(eligible)
        return [True, False]

    assert check_answer(sentences, judge) == [True, False]
    assert seen == [sentences]


def test_missing_verdict_counts_as_unsupported() -> None:
    assert check_answer(
        [
            ("First supported claim.", ["First supported claim."]),
            ("Second supported claim.", ["Second supported claim."]),
        ],
        lambda _: [True],
    ) == [True, False]


def test_no_judge_call_when_nothing_passes_deterministic_checks() -> None:
    calls = []

    def judge(sentences: list[tuple[str, list[str]]]) -> list[bool]:
        calls.append(sentences)
        return []

    result = check_answer(
        [("The total was $30.", ["The total was $20."]), ("No sources.", [])],
        judge,
    )

    assert result == [False, False]
    assert calls == []
