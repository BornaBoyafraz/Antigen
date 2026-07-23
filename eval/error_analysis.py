"""Train the classifier and expose its held-out mistakes as grouped JSON.

The report includes both the stratified held-out split and the separately
authored adversarial suite. Grouping every miss by category makes recurring
failure modes inspectable without hiding which evaluation set produced them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from sklearn.pipeline import Pipeline

from eval.harness import load_adversarial_suite, split_dataset
from model import Example, load_dataset, predict_one, train


class MisclassifiedExample(TypedDict):
    text: str
    true_label: str
    predicted_label: str
    score: float
    category: str
    evaluation_set: str


ErrorGroups = dict[str, list[MisclassifiedExample]]


def collect_misclassifications(
    pipeline: Pipeline,
    examples: Sequence[Example],
    evaluation_set: str,
) -> ErrorGroups:
    """Return every incorrect prediction grouped by the example category."""
    grouped: ErrorGroups = {}
    for example in examples:
        predicted_label, score = predict_one(pipeline, example.text)
        if predicted_label == example.label:
            continue

        record: MisclassifiedExample = {
            "text": example.text,
            "true_label": example.label,
            "predicted_label": predicted_label,
            "score": score,
            "category": example.category,
            "evaluation_set": evaluation_set,
        }
        grouped.setdefault(example.category, []).append(record)
    return grouped


def _merge_groups(*reports: ErrorGroups) -> ErrorGroups:
    merged: ErrorGroups = {}
    for report in reports:
        for category, records in report.items():
            merged.setdefault(category, []).extend(records)
    return {category: merged[category] for category in sorted(merged)}


def run_error_analysis() -> ErrorGroups:
    """Train on the standard split and collect held-out and adversarial misses."""
    training_examples, held_out_examples = split_dataset(load_dataset())
    pipeline = train(training_examples)
    return _merge_groups(
        collect_misclassifications(pipeline, held_out_examples, "held_out"),
        collect_misclassifications(
            pipeline,
            load_adversarial_suite(),
            "adversarial_suite",
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="write JSON to this path instead of standard output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the error report and write it to a file or standard output."""
    args = _parse_args(argv)
    report = run_error_analysis()
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        error_count = sum(len(records) for records in report.values())
        print(f"Wrote {error_count} misclassified examples to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
