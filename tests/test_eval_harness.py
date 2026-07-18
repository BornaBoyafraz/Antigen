from eval.harness import format_report, load_adversarial_suite, run_eval


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


def test_format_report_is_a_nonempty_string():
    results = run_eval()
    report = format_report(results)
    assert isinstance(report, str)
    assert "Held-out test split" in report
    assert "Adversarial suite" in report
