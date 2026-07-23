"""Tests for engineered-feature coefficient inspection."""
from __future__ import annotations

from eval.feature_importance import extract_engineered_coefficients, format_report
from features import FeatureResult
from model import load_dataset, train


def test_engineered_coefficients_are_named_and_sorted_by_magnitude() -> None:
    report = extract_engineered_coefficients(train(load_dataset()))
    rows = report.coefficients

    assert report.n_estimators == 3
    assert {row["feature"] for row in rows} == set(FeatureResult.vector_names())
    assert [row["magnitude"] for row in rows] == sorted(
        (row["magnitude"] for row in rows),
        reverse=True,
    )
    assert all(row["magnitude"] == abs(row["coefficient"]) for row in rows)
    assert {row["direction"] for row in rows} <= {
        "toward injection",
        "toward benign",
        "neutral",
    }


def test_feature_importance_report_explains_averaging_and_direction() -> None:
    report = extract_engineered_coefficients(train(load_dataset()))

    rendered = format_report(report)

    assert "Mean across 3 calibrated logistic-regression fold estimators." in rendered
    assert "Positive values move the base logit toward injection" in rendered
    assert all(feature in rendered for feature in FeatureResult.vector_names())
