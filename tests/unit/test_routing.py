from minutes.llm.routing import is_easy_batch, is_easy_question


def test_easy_question_needs_score_at_least_080() -> None:
    assert is_easy_question("What was approved?", 0.80)
    assert not is_easy_question("What was approved?", 0.79)


def test_multi_step_words_are_never_easy() -> None:
    for question in (
        "How has the council voted on zoning since 2023?",
        "List every contract",
        "How many times was this discussed?",
        "compare the two proposals",
    ):
        assert not is_easy_question(question, 0.95)


def test_word_match_is_whole_word_and_case_insensitive() -> None:
    assert is_easy_question("Does the ball fit?", 0.95)
    assert not is_easy_question("Does ALL fit?", 0.95)


def test_easy_batch_has_no_vote_label_and_short_items() -> None:
    assert not is_easy_batch(["Ayes: 4 Noes: 2"])
    assert not is_easy_batch(["x" * 4001])
    assert is_easy_batch(["Short item one", "Short item two"])
