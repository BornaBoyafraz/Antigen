"""Report learned logistic-regression weights for engineered features.

The scoring model calibrates three fold-specific logistic regressions. This
script averages their coefficients, then sorts the engineered signals by
absolute magnitude so their learned direction and relative weighting can be
inspected without mixing in thousands of character n-grams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from sklearn.pipeline import Pipeline

from features import FeatureResult
from model import LABEL_TO_INT, load_dataset, train


class FeatureCoefficient(TypedDict):
    feature: str
    coefficient: float
    magnitude: float
    direction: str


@dataclass(frozen=True)
class FeatureImportanceReport:
    coefficients: list[FeatureCoefficient]
    n_estimators: int


def extract_engineered_coefficients(pipeline: Pipeline) -> FeatureImportanceReport:
    """Average calibrated-fold weights and return engineered features by magnitude."""
    feature_union = pipeline.named_steps["features"]
    calibrated_classifier = pipeline.named_steps["clf"]
    feature_names = [str(name) for name in feature_union.get_feature_names_out()]
    feature_indexes = {name: index for index, name in enumerate(feature_names)}

    calibrated_folds = getattr(calibrated_classifier, "calibrated_classifiers_", None)
    if not calibrated_folds:
        raise ValueError("pipeline classifier has not been fitted")

    coefficient_rows: list[np.ndarray] = []
    for calibrated_fold in calibrated_folds:
        estimator = getattr(calibrated_fold, "estimator", None)
        if estimator is None:
            estimator = getattr(calibrated_fold, "base_estimator", None)
        if estimator is None or not hasattr(estimator, "coef_"):
            raise TypeError("calibrated fold does not expose logistic-regression coefficients")
        if list(estimator.classes_) != [LABEL_TO_INT["benign"], LABEL_TO_INT["injection"]]:
            raise ValueError("unexpected classifier class order")

        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.shape != (1, len(feature_names)):
            raise ValueError("unexpected logistic-regression coefficient shape")
        coefficient_rows.append(coefficients[0])

    mean_coefficients = np.mean(np.vstack(coefficient_rows), axis=0)
    rows: list[FeatureCoefficient] = []
    for feature in FeatureResult.vector_names():
        qualified_name = f"heuristics__{feature}"
        if qualified_name not in feature_indexes:
            raise ValueError(f"engineered feature is missing from fitted pipeline: {feature}")
        coefficient = float(mean_coefficients[feature_indexes[qualified_name]])
        if coefficient > 0:
            direction = "toward injection"
        elif coefficient < 0:
            direction = "toward benign"
        else:
            direction = "neutral"
        rows.append(
            {
                "feature": feature,
                "coefficient": coefficient,
                "magnitude": abs(coefficient),
                "direction": direction,
            }
        )

    rows.sort(key=lambda row: (-row["magnitude"], row["feature"]))
    return FeatureImportanceReport(coefficients=rows, n_estimators=len(coefficient_rows))


def format_report(report: FeatureImportanceReport) -> str:
    """Format coefficient direction and magnitude as a plain-text table."""
    lines = [
        "Engineered feature coefficients",
        f"Mean across {report.n_estimators} calibrated logistic-regression fold estimators.",
        "Positive values move the base logit toward injection; negative values toward benign.",
        "",
        f"{'feature':<36} {'coefficient':>12} {'magnitude':>12}  direction",
        f"{'-' * 36} {'-' * 12} {'-' * 12}  {'-' * 16}",
    ]
    for row in report.coefficients:
        lines.append(
            f"{row['feature']:<36} {row['coefficient']:>12.6f} "
            f"{row['magnitude']:>12.6f}  {row['direction']}"
        )
    return "\n".join(lines)


def main() -> int:
    """Train the standard pipeline and print its engineered-feature weights."""
    pipeline = train(load_dataset())
    print(format_report(extract_engineered_coefficients(pipeline)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
