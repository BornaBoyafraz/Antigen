"""A transparent, non-learned baseline classifier.

Its entire purpose is to answer the question every classifier project should
answer but most skip: does the trained model actually beat the naive thing you
could write in an afternoon? If a char-n-gram TF-IDF + calibrated logistic
regression can't outperform pure keyword matching, the ML isn't earning its
complexity. The eval harness scores this side by side with the real model so the
"the model adds value" claim is checkable, not asserted.

The baseline reuses the exact same engineered signal families as `features.py`
(instruction-override phrases, spoofed role markers, text addressed at the
assistant, indirect-content framing, zero-width obfuscation) but with a fixed
rule instead of learned weights and, crucially, no char-n-gram TF-IDF — so it
has no way to generalize to phrasings that don't trip a literal regex. That gap
is exactly where the trained model is expected to pull ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

from features import extract_features
from model import Example


@dataclass
class RegexBaselineClassifier:
    """Flags injection from unambiguous signals that survive discussion/quote
    context. No training, no learned weights — deliberately the dumb baseline."""

    def predict_one(self, text: str) -> tuple[str, float]:
        feats = extract_features(text)

        # "Bare" trigger hits: override phrases + spoofed role markers + text
        # addressed at the assistant, minus the ones that are quoted or framed
        # as discussion rather than issued as an instruction.
        bare = (
            feats.override_phrase_count
            + feats.role_spoof_count
            + feats.addressed_to_assistant_count
            - feats.discussion_context_count
        )

        # Integer count of fired strong signals. Kept as a saturating score via
        # s / (s + 1): 0 -> 0.0, 1 -> 0.5, larger -> approaches 1.0. Monotonic,
        # so ROC-AUC is meaningful, and the 0.5 decision threshold lands exactly
        # at "at least one strong signal fired."
        strength = float(
            max(0, bare)
            + feats.indirect_frame_marker_count
            + (1 if feats.zero_width_char_count else 0)
        )
        score = strength / (strength + 1.0)
        label = "injection" if score >= 0.5 else "benign"
        return label, score


def fit_baseline(examples: list[Example] | None = None) -> RegexBaselineClassifier:
    """No-op fit, present only for interface symmetry with `model.train()` so
    the eval harness can treat the baseline like the trained pipeline. The
    baseline is rule-based and has nothing to learn from the data."""
    return RegexBaselineClassifier()
