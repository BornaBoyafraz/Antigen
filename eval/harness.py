"""Evaluation harness: stratified train/test split with per-category
precision/recall/F1, a confusion matrix, and a held-out "adversarial suite"
of harder, hand-written phrasings the model never trained on — the
generalization check that a random split of the same templates can't give
you, since a random split still leaks the templates' phrasing into training.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from baselines import fit_baseline
from model import LABEL_TO_INT, Example, load_dataset, predict_one, train

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


PredictFn = Callable[[str], tuple[str, float]]


def _score_examples(predict_fn: PredictFn, examples: list[Example]) -> dict:
    y_true = [LABEL_TO_INT[e.label] for e in examples]
    y_pred = []
    y_score = []
    for e in examples:
        label, score = predict_fn(e.text)
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
    for e, pred_int in zip(examples, y_pred, strict=True):
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

    # The trained model uses predict_one(pipeline, text); the baseline exposes
    # its own predict_one(text). Wrap both to the same text -> (label, score)
    # shape so _score_examples is model-agnostic.
    def model_predict(text: str) -> tuple[str, float]:
        return predict_one(pipeline, text)

    baseline = fit_baseline(train_ex)
    baseline_predict: PredictFn = baseline.predict_one

    adversarial = load_adversarial_suite()

    return {
        "n_train": len(train_ex),
        "n_test": len(test_ex),
        "held_out": _score_examples(model_predict, test_ex),
        "adversarial_suite": _score_examples(model_predict, adversarial),
        "baseline_held_out": _score_examples(baseline_predict, test_ex),
        "baseline_adversarial_suite": _score_examples(baseline_predict, adversarial),
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

    if "baseline_held_out" in results:
        lines.append("=== Model vs. regex-only baseline (does the ML earn its keep?) ===")
        lines.append(f"  {'split':<28} {'model acc':>10} {'baseline acc':>14} {'lift':>8}")
        for label, model_key, base_key in [
            ("held-out", "held_out", "baseline_held_out"),
            ("adversarial (unseen)", "adversarial_suite", "baseline_adversarial_suite"),
        ]:
            m_acc = results[model_key]["classification_report"]["accuracy"]
            b_acc = results[base_key]["classification_report"]["accuracy"]
            lines.append(
                f"  {label:<28} {m_acc:>10.3f} {b_acc:>14.3f} {m_acc - b_acc:>+8.3f}"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    results = run_eval()
    print(format_report(results))
