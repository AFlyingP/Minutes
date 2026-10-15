import pytest

from minutes.retrieval.search import rrf


def test_rrf_scores_and_order() -> None:
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert [chunk_id for chunk_id, _ in fused] == [1, 3, 2, 4]
    assert dict(fused) == pytest.approx(
        {1: 1 / 61 + 1 / 62, 3: 1 / 63 + 1 / 61, 2: 1 / 62, 4: 1 / 63}
    )


def test_rrf_tie_breaks_by_id() -> None:
    assert rrf([[9, 2], [2, 9]]) == [(2, 1 / 61 + 1 / 62), (9, 1 / 61 + 1 / 62)]


def test_rrf_single_list_keeps_order() -> None:
    assert rrf([[9, 2, 5]], c=10) == [(9, 1 / 11), (2, 1 / 12), (5, 1 / 13)]


def test_rrf_empty_lists() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []
    assert rrf([[], [3], []]) == [(3, 1 / 61)]
