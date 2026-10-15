from minutes.evals.gate import GATED, compare, extract_gated


def test_metric_below_baseline_minus_tolerance_fails() -> None:
    baseline = {"answer_accuracy": 0.80}
    assert (
        len(compare({"answers": {"final": {"answer_accuracy": {"median": 0.76}}}}, baseline)) == 1
    )
    assert compare({"answers": {"final": {"answer_accuracy": {"median": 0.77}}}}, baseline) == []


def test_each_gated_metric_has_its_tolerance() -> None:
    assert {name: value[2] for name, value in GATED.items()} == {
        "recall_at_10": 0.02,
        "answer_accuracy": 0.03,
        "faithfulness": 0.03,
        "decline_accuracy": 0.084,
        "extraction_accuracy": 0.02,
        "segmentation_exact_rate": 0.05,
        "agent_completion": 0.05,
        "cost_per_query_usd": 0.10,
    }


def test_cost_fails_when_above_110_percent() -> None:
    baseline = {"cost_per_query_usd": 0.010}
    assert len(compare({"cost": {"controls_on": {"cost_per_query_usd": 0.0111}}}, baseline)) == 1
    assert compare({"cost": {"controls_on": {"cost_per_query_usd": 0.0110}}}, baseline) == []


def test_metric_missing_from_baseline_is_not_compared() -> None:
    results = {"segmentation": {"overall": {"exact_rate": 0.0}}}
    assert compare(results, {}) == []


def test_metric_null_in_results_fails_when_in_baseline() -> None:
    results = {"segmentation": {"overall": {"exact_rate": None}}}
    assert len(compare(results, {"segmentation_exact_rate": 0.9})) == 1


def test_extract_gated_reads_documented_paths() -> None:
    results = {
        "retrieval": {"item": {"hybrid_rerank": {"recall_at_10": 0.81}}},
        "answers": {
            "final": {
                "answer_accuracy": {"median": 0.82},
                "faithfulness": {"median": 0.83},
                "decline_accuracy": {"median": 0.84},
            }
        },
        "extraction": {"overall": {"median": 0.85}},
        "segmentation": {"overall": {"exact_rate": 0.86}},
        "agent": {"completion": {"median": 0.87}},
        "cost": {"controls_on": {"cost_per_query_usd": 0.012}},
    }
    assert extract_gated(results) == {
        "recall_at_10": 0.81,
        "answer_accuracy": 0.82,
        "faithfulness": 0.83,
        "decline_accuracy": 0.84,
        "extraction_accuracy": 0.85,
        "segmentation_exact_rate": 0.86,
        "agent_completion": 0.87,
        "cost_per_query_usd": 0.012,
    }
