from dataclasses import replace
from datetime import date
from typing import cast

from minutes.evals.metrics import (
    cohen_kappa,
    mean_or_none,
    passage_retrieved,
    percentile,
    recall_for_question,
    segmentation_metrics,
    summarize_runs,
)
from minutes.labels.schema import Passage
from minutes.retrieval.search import Hit, Span


def _passage(document_id: str, unit_index: int, start: int, end: int) -> Passage:
    return Passage(
        document_id=document_id,
        unit_index=unit_index,
        start=start,
        end=end,
        text="source text",
    )


def _span(document_id: str, unit_index: int, start: int, end: int) -> Span:
    return Span(
        chunk_id=1,
        document_id=document_id,
        unit_index=unit_index,
        unit_kind="page",
        start_offset=start,
        end_offset=end,
        text="retrieved text",
    )


def _hit(rank: int, spans: list[Span]) -> Hit:
    return Hit(
        rank=rank,
        score=1.0,
        chunk_id=rank,
        item_id=None,
        document_id="alder-minutes-101",
        city_id="alder",
        body="City Council",
        meeting_date=date(2024, 2, 6),
        doc_kind="minutes",
        item_identifier=None,
        item_title=None,
        start_ms=None,
        spans=spans,
    )


def test_passage_retrieved_needs_half_coverage() -> None:
    passage = _passage("minutes-1", 2, 100, 200)
    assert passage_retrieved(passage, [_span("minutes-1", 2, 150, 400)])
    assert not passage_retrieved(passage, [_span("minutes-1", 2, 151, 400)])


def test_coverage_is_union_of_spans() -> None:
    passage = _passage("minutes-1", 2, 100, 200)
    assert passage_retrieved(
        passage,
        [_span("minutes-1", 2, 100, 130), _span("minutes-1", 2, 170, 200)],
    )
    assert passage_retrieved(
        passage,
        [_span("minutes-1", 2, 100, 160), _span("minutes-1", 2, 140, 200)],
    )


def test_span_in_other_unit_or_document_does_not_count() -> None:
    passage = _passage("minutes-1", 2, 100, 200)
    assert not passage_retrieved(passage, [_span("minutes-1", 3, 100, 200)])
    assert not passage_retrieved(passage, [_span("minutes-2", 2, 100, 200)])


def test_recall_uses_offsets_only() -> None:
    passage = _passage("minutes-1", 2, 100, 200)
    span = _span("minutes-1", 2, 100, 160)
    without_text = passage.model_copy(update={"text": None})
    span_without_text = replace(span, text=cast(str, None))
    assert recall_for_question([passage], [_hit(1, [span])], 5) == recall_for_question(
        [without_text], [_hit(1, [span_without_text])], 5
    )


def test_recall_counts_only_top_k_hits() -> None:
    passage = _passage("alder-minutes-101", 2, 100, 200)
    hits = [_hit(rank, []) for rank in range(1, 6)]
    hits.append(_hit(6, [_span("alder-minutes-101", 2, 100, 200)]))
    assert recall_for_question([passage], hits, 5) == 0.0
    assert recall_for_question([passage], hits, 10) == 1.0


def test_recall_is_fraction_of_gold_passages() -> None:
    passages = [_passage("minutes-1", 1, 0, 20), _passage("minutes-1", 2, 0, 20)]
    hits = [_hit(1, [_span("minutes-1", 1, 0, 20)])]
    assert recall_for_question(passages, hits, 5) == 0.5


def test_mean_or_none_empty_is_none() -> None:
    assert mean_or_none([]) is None


def test_segmentation_metrics() -> None:
    result = segmentation_metrics([("m1", 4, 4), ("m2", 2, 3), ("m3", 5, 5), ("m4", 2, 0)])
    assert result["exact_rate"] == 0.5
    assert result["mae"] == 0.75
    assert result["signed_errors"] == [0, 1, 0, -2]


def test_percentile_nearest_rank() -> None:
    values = [float(value) for value in range(1, 21)]
    assert percentile(values, 50) == 10
    assert percentile(values, 95) == 19


def test_summarize_runs_min_median_max() -> None:
    assert summarize_runs([0.1, 0.4, 0.2]) == {
        "runs": 3,
        "min": 0.1,
        "median": 0.2,
        "max": 0.4,
    }
    assert summarize_runs([1.0, 3.0])["median"] == 2.0


def test_summarize_runs_with_none_is_none() -> None:
    assert summarize_runs([0.1, None, 0.9]) == {
        "runs": 3,
        "min": None,
        "median": None,
        "max": None,
    }


def test_cohen_kappa_known_value() -> None:
    assert cohen_kappa([True, True, False, False], [True, True, False, True]) == 0.5


def test_cohen_kappa_degenerate_denominator() -> None:
    assert cohen_kappa([True, True], [True, True]) == 1.0


def test_recall_zero_denominators_propagate_null() -> None:
    empty = _passage("minutes-1", 1, 10, 10)
    assert passage_retrieved(empty, []) is None
    assert recall_for_question([empty], [], 5) is None
    assert recall_for_question([], [], 5) is None
    assert mean_or_none([1.0, recall_for_question([empty], [], 5)]) is None
    assert mean_or_none([1.0, recall_for_question([], [], 5)]) is None


def test_percentile_empty_is_none() -> None:
    assert percentile([], 50) is None
    assert percentile([], 95) is None


def test_cohen_kappa_without_cases_is_none() -> None:
    assert cohen_kappa([], []) is None
