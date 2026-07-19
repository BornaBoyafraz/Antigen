"""Human-readable rationale for a classification.

The primary pipeline in model.py uses CalibratedClassifierCV for
well-calibrated probabilities, which internally fits several classifier
copies (one per CV fold) — convenient for scoring, awkward to introspect
for a single clean coefficient vector. So explanation uses its own plain
(uncalibrated) LogisticRegression fitted on the same feature union: same
inputs, same signal, but a single coefficient vector that's actually
meant to be read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from features import extract_features
from model import LABEL_TO_INT, Example, HeuristicFeatureTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

TOP_NGRAMS = 5


@dataclass
class Explanation:
    label: str
    score: float
    triggered_phrases: list[str] = field(default_factory=list)
    triggered_role_markers: list[str] = field(default_factory=list)
    triggered_addressed_phrases: list[str] = field(default_factory=list)
    triggered_indirect_frames: list[str] = field(default_factory=list)
    discussion_context_phrases: list[str] = field(default_factory=list)
    heuristic_summary: dict[str, float] = field(default_factory=dict)
    top_contributing_ngrams: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "triggered_phrases": self.triggered_phrases,
            "triggered_role_markers": self.triggered_role_markers,
            "triggered_addressed_phrases": self.triggered_addressed_phrases,
            "triggered_indirect_frames": self.triggered_indirect_frames,
            "discussion_context_phrases": self.discussion_context_phrases,
            "heuristic_summary": self.heuristic_summary,
            "top_contributing_ngrams": [
                {"ngram": ng, "weight": w} for ng, w in self.top_contributing_ngrams
            ],
        }


def build_explainer_pipeline() -> Pipeline:
    union = FeatureUnion([
        ("tfidf_char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=4000)),
        ("heuristics", HeuristicFeatureTransformer()),
    ])
    return Pipeline([
        ("features", union),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def fit_explainer(examples: list[Example]) -> Pipeline:
    X = [e.text for e in examples]
    y = [LABEL_TO_INT[e.label] for e in examples]
    pipeline = build_explainer_pipeline()
    pipeline.fit(X, y)
    return pipeline


def _top_ngrams_for_text(pipeline: Pipeline, text: str, k: int = TOP_NGRAMS) -> list[tuple[str, float]]:
    union: FeatureUnion = pipeline.named_steps["features"]
    tfidf: TfidfVectorizer = dict(union.transformer_list)["tfidf_char"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    vec = tfidf.transform([text])
    vocab = tfidf.get_feature_names_out()
    n_tfidf_features = len(vocab)
    coefs = clf.coef_[0][:n_tfidf_features]

    scored = []
    coo = vec.tocoo()
    for col, val in zip(coo.col, coo.data):
        contribution = val * coefs[col]
        scored.append((vocab[col], float(contribution)))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [pair for pair in scored[:k] if pair[1] > 0]


def explain(pipeline: Pipeline, explainer_pipeline: Pipeline, text: str) -> Explanation:
    from model import predict_one

    label, score = predict_one(pipeline, text)
    feats = extract_features(text)

    return Explanation(
        label=label,
        score=score,
        triggered_phrases=feats.matched_override_phrases,
        triggered_role_markers=feats.matched_role_markers,
        triggered_addressed_phrases=feats.matched_addressed_phrases,
        triggered_indirect_frames=feats.matched_indirect_frames,
        discussion_context_phrases=feats.matched_discussion_context_phrases,
        heuristic_summary={
            "imperative_density": round(feats.imperative_density, 4),
            "base64_span_ratio": round(feats.base64_span_ratio, 4),
            "hex_or_url_escape_count": feats.hex_or_url_escape_count,
            "zero_width_char_count": feats.zero_width_char_count,
            "uppercase_ratio": round(feats.uppercase_ratio, 4),
            "discussion_context_count": feats.discussion_context_count,
        },
        top_contributing_ngrams=_top_ngrams_for_text(explainer_pipeline, text),
    )
