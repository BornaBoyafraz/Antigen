"""The classifier: char-n-gram TF-IDF unioned with engineered heuristic
features (features.py), feeding a calibrated logistic regression.

Logistic regression over an interpretable feature union is the deliberate
choice here, not gradient boosting or a fine-tuned transformer: every
prediction can be attributed to specific, human-readable signals (see
explain.py), which matters more for a security classifier than a couple
points of accuracy from a black-box model. See docs/ARCHITECTURE.md for
the full rationale and the documented path to swapping in a heavier model
behind the same interface.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from features import FeatureResult, extract_features

DATA_PATH = Path(__file__).parent / "data" / "prompts.jsonl"

LABELS = ["benign", "injection"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}


@dataclass
class Example:
    text: str
    label: str
    category: str


def load_dataset(path: Path = DATA_PATH) -> list[Example]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(Example(text=row["text"], label=row["label"], category=row["category"]))
    return examples


class HeuristicFeatureTransformer(BaseEstimator, TransformerMixin):
    """Wraps features.extract_features as a scikit-learn transformer so it
    can sit in a FeatureUnion next to the TF-IDF vectorizer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectors = [extract_features(text).to_vector() for text in X]
        return np.asarray(vectors, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FeatureResult.vector_names())


def build_pipeline() -> Pipeline:
    union = FeatureUnion([
        ("tfidf_char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=4000)),
        ("heuristics", HeuristicFeatureTransformer()),
    ])
    base_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    return Pipeline([
        ("features", union),
        ("clf", calibrated),
    ])


def train(examples: list[Example]) -> Pipeline:
    X = [e.text for e in examples]
    y = [LABEL_TO_INT[e.label] for e in examples]
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline


def predict_one(pipeline: Pipeline, text: str) -> tuple[str, float]:
    """Returns (label, probability_of_injection)."""
    proba = pipeline.predict_proba([text])[0]
    injection_idx = LABEL_TO_INT["injection"]
    score = float(proba[injection_idx])
    label = "injection" if score >= 0.5 else "benign"
    return label, score
