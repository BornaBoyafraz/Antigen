"""Evaluation harness: stratified train/test split with per-category
precision/recall/F1, a confusion matrix, and a held-out "adversarial suite"
of harder, hand-written phrasings the model never trained on — the
generalization check that a random split of the same templates can't give
you, since a random split still leaks the templates' phrasing into training.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from model import Example, LABEL_TO_INT, load_dataset, predict_one, train

ADVERSARIAL_SUITE_PATH = Path(__file__).parent / "adversarial_suite.jsonl"


def load_adversarial_suite(path: Path = ADVERSARIAL_SUITE_PATH) -> list[Example]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(Example(text=row["text"], label=row["label"], category=row["category"]))
    return examples


def split_dataset(examples: list[Example], test_size: float = 0.25, seed: int = 42):
    labels = [e.label for e in examples]
    train_ex, test_ex = train_test_split(
        examples, test_size=test_size, random_state=seed, stratify=labels
    )
    return train_ex, test_ex


def _score_examples(pipeline, examples: list[Example]) -> dict:
    y_true = [LABEL_TO_INT[e.label] for e in examples]
    y_pred = []
    y_score = []
    for e in examples:
        label, score = predict_one(pipeline, e.text)
        y_pred.append(LABEL_TO_INT[label])
        y_score.append(score)

    report = classification_report(
        y_true, y_pred, target_names=["benign", "injection"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    by_category: dict[str, dict] = {}
    for e, pred_int in zip(examples, y_pred):
        cat = by_category.setdefault(e.category, {"n": 0, "correct": 0})
        cat["n"] += 1
        cat["correct"] += int(pred_int == LABEL_TO_INT[e.label])
    category_accuracy = {
        cat: round(stats["correct"] / stats["n"], 4) for cat, stats in by_category.items()
    }

    return {
        "n": len(examples),
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": auc,
        "category_accuracy": category_accuracy,
    }


def run_eval() -> dict:
    all_examples = load_dataset()
    train_ex, test_ex = split_dataset(all_examples)
    pipeline = train(train_ex)

    held_out_results = _score_examples(pipeline, test_ex)

    adversarial = load_adversarial_suite()
    adversarial_results = _score_examples(pipeline, adversarial)

    return {
        "n_train": len(train_ex),
        "n_test": len(test_ex),
        "held_out": held_out_results,
        "adversarial_suite": adversarial_results,
    }


def format_report(results: dict) -> str:
    lines = []
    lines.append(f"Train examples: {results['n_train']}   Test examples: {results['n_test']}")
    lines.append("")

    for section_name, key in [("Held-out test split", "held_out"), ("Adversarial suite (unseen phrasing)", "adversarial_suite")]:
        section = results[key]
        report = section["classification_report"]
        lines.append(f"=== {section_name} (n={section['n']}) ===")
        lines.append(
            f"  {'label':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}"
        )
        for label in ["benign", "injection"]:
            r = report[label]
            lines.append(
                f"  {label:<10} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f} {int(r['support']):>8}"
            )
        lines.append(f"  accuracy: {report['accuracy']:.3f}   roc_auc: {section['roc_auc']:.3f}")
        lines.append(f"  confusion matrix [[TN, FP], [FN, TP]]: {section['confusion_matrix']}")
        lines.append("  per-category accuracy:")
        for cat, acc in sorted(section["category_accuracy"].items()):
            lines.append(f"    {cat:<24} {acc:.3f}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    results = run_eval()
    print(format_report(results))
