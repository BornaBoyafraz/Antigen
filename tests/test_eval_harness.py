import pytest

from eval.harness import (
    compute_threshold_metrics,
    format_report,
    load_adversarial_suite,
    run_eval,
)


def test_adversarial_suite_loads_and_has_both_labels():
    examples = load_adversarial_suite()
    labels = {e.label for e in examples}
    assert labels == {"benign", "injection"}
    assert len(examples) >= 20


def test_run_eval_produces_expected_shape():
    results = run_eval()
    assert "held_out" in results
    assert "adversarial_suite" in results
    for key in ("held_out", "adversarial_suite"):
        section = results[key]
        assert "classification_report" in section
        assert "confusion_matrix" in section
        assert "category_accuracy" in section
        assert len(section["threshold_metrics"]) == 9


def test_compute_threshold_metrics_captures_operating_point_tradeoff():
    points = compute_threshold_metrics(
        y_true=[0, 0, 1, 1],
        y_score=[0.2, 0.6, 0.4, 0.8],
        thresholds=(0.3, 0.7),
    )

    assert points[0] == {
        "threshold": 0.3,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": pytest.approx(0.8),
        "false_positive_rate": 0.5,
    }
    assert points[1] == {
        "threshold": 0.7,
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "false_positive_rate": 0.0,
    }


def test_compute_threshold_metrics_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same length"):
        compute_threshold_metrics([0, 1], [0.2])


def test_format_report_is_a_nonempty_string():
    results = run_eval()
    report = format_report(results)
    assert isinstance(report, str)
    assert "Held-out test split" in report
    assert "Adversarial suite" in report
    assert "false-positive rate" in report
