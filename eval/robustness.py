"""Adversarial robustness evaluation: how well does detection survive evasion?

Clean-data accuracy is the wrong headline number for a security classifier.
An attacker who wants their injection through will not submit the canonical
phrasing the model trained on — they will obfuscate it. So the question that
actually matters for a defensive filter is: of the attacks the model catches
in the clear, how many does it *still* catch once a motivated adversary
applies cheap, well-known evasion transforms?

This module answers that quantitatively. It takes the injection examples the
model already flags, applies a battery of evasion perturbations (zero-width
insertion, homoglyph substitution, intra-word spacing, case flipping,
leetspeak), and reports the *retention rate* per transform — the fraction of
originally-caught attacks that remain caught after perturbation. A low
retention rate is an evasion that defeats the detector, and naming it is more
useful than a single accuracy figure that hides it.

The transforms are deliberately character-level and semantics-preserving: the
perturbed text still reads as the same attack to a human (and to the LLM it
targets), so a drop in detection is a genuine robustness gap, not the model
correctly ignoring now-benign text.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from eval.harness import load_adversarial_suite
from model import Example, load_dataset, predict_one, train

# Zero-width characters an attacker can splice between the letters of a
# trigger word: invisible to a human, but they break the literal substrings a
# word-level or naive matcher keys on.
_ZERO_WIDTH_CHARS = ["​", "‌", "‍", "⁠"]

# Latin -> confusable (Cyrillic/Greek) homoglyphs. Visually near-identical,
# different code points.
_HOMOGLYPHS = {
    "a": "а",  # Cyrillic a
    "e": "е",  # Cyrillic e
    "o": "о",  # Cyrillic o
    "p": "р",  # Cyrillic er
    "c": "с",  # Cyrillic es
    "i": "і",  # Cyrillic byelorussian-ukrainian i
    "y": "у",  # Cyrillic u
}

_LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


def _perturb_letters(text: str, rng: random.Random, transform: Callable[[str], str], rate: float) -> str:
    """Apply a per-letter transform to a random `rate` fraction of alphabetic
    characters, leaving structure and non-letters intact."""
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < rate:
            out.append(transform(ch))
        else:
            out.append(ch)
    return "".join(out)


def zero_width_insertion(text: str, rng: random.Random) -> str:
    """Splice an invisible zero-width character after ~half of the letters."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < 0.5:
            out.append(rng.choice(_ZERO_WIDTH_CHARS))
    return "".join(out)


def homoglyph_substitution(text: str, rng: random.Random) -> str:
    return _perturb_letters(text, rng, lambda c: _HOMOGLYPHS.get(c.lower(), c), rate=0.6)


def intra_word_spacing(text: str, rng: random.Random) -> str:
    """Insert a space inside words so trigger substrings no longer match."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < 0.25:
            out.append(" ")
    return "".join(out)


def case_flip(text: str, rng: random.Random) -> str:
    return _perturb_letters(text, rng, lambda c: c.upper() if c.islower() else c.lower(), rate=0.5)


def leetspeak(text: str, rng: random.Random) -> str:
    return _perturb_letters(text, rng, lambda c: _LEET.get(c.lower(), c), rate=0.7)


EVASIONS: dict[str, Callable[[str, random.Random], str]] = {
    "zero_width_insertion": zero_width_insertion,
    "homoglyph_substitution": homoglyph_substitution,
    "intra_word_spacing": intra_word_spacing,
    "case_flip": case_flip,
    "leetspeak": leetspeak,
}


@dataclass
class EvasionResult:
    name: str
    caught_baseline: int  # attacks caught before perturbation
    caught_after: int  # of those, still caught after
    retention_rate: float  # caught_after / caught_baseline (1.0 = fully robust)


def evaluate_robustness(
    pipeline,
    injections: list[Example],
    threshold: float = 0.5,
    seed: int = 0,
) -> list[EvasionResult]:
    """For each evasion, measure the retention rate over the injection
    examples the model catches in the clear.

    Only examples scored >= threshold at baseline count toward the
    denominator: an evasion can only be blamed for hiding an attack the
    detector would otherwise have caught.
    """
    baseline_caught: list[Example] = [
        ex for ex in injections if predict_one(pipeline, ex.text)[1] >= threshold
    ]

    results: list[EvasionResult] = []
    for name, transform in EVASIONS.items():
        rng = random.Random(seed)
        still = 0
        for ex in baseline_caught:
            perturbed = transform(ex.text, rng)
            if predict_one(pipeline, perturbed)[1] >= threshold:
                still += 1
        n = len(baseline_caught)
        results.append(
            EvasionResult(
                name=name,
                caught_baseline=n,
                caught_after=still,
                retention_rate=(still / n) if n else 0.0,
            )
        )
    return results


def _injection_examples() -> list[Example]:
    pool = load_dataset() + load_adversarial_suite()
    return [ex for ex in pool if ex.label == "injection"]


def format_report(results: list[EvasionResult]) -> str:
    lines = ["=== Adversarial robustness (evasion retention) ==="]
    lines.append(f"  {'evasion':<24} {'baseline':>9} {'retained':>9} {'retention':>10}")
    for r in results:
        lines.append(
            f"  {r.name:<24} {r.caught_baseline:>9} {r.caught_after:>9} {r.retention_rate:>9.1%}"
        )
    if results:
        weakest = min(results, key=lambda r: r.retention_rate)
        lines.append("")
        lines.append(
            f"  Weakest against: {weakest.name} "
            f"(retains {weakest.retention_rate:.1%} of caught attacks)"
        )
    return "\n".join(lines)


def run() -> list[EvasionResult]:
    pipeline = train(load_dataset())
    return evaluate_robustness(pipeline, _injection_examples())


if __name__ == "__main__":
    print(format_report(run()))
