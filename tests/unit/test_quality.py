from minutes.ingest.quality import levenshtein, valid_token_ratio


def test_valid_token_ratio_examples() -> None:
    assert valid_token_ratio("The Council adopted Resolution 2024-07 for $184,500.") == 1.0
    assert valid_token_ratio("The xqzt ~~ vote") == 0.5
    assert valid_token_ratio("") == 0.0
    assert valid_token_ratio(" \t\n., [] | * ") == 0.0
    assert valid_token_ratio("Council's Council\u2019s land-use 65ft 7:15 CB3001 A") == 1.0
    assert valid_token_ratio("LLC xqzt 12abc abc123 123x-y") == 0.0


def test_levenshtein_known_values() -> None:
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("same text", "same text") == 0
    assert levenshtein("", "") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("sitting", "kitten") == 3
