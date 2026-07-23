"""Tests for complete, category-grouped model error reports."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from eval import error_analysis
from model import Example


def test_collect_misclassifications_groups_every_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = [
        Example(text="false positive", label="benign", category="benign_hard"),
        Example(text="correct injection", label="injection", category="direct_injection"),
        Example(text="false negative", label="injection", category="direct_injection"),
    ]
    predictions = {
        "false positive": ("injection", 0.81),
        "correct injection": ("injection", 0.92),
        "false negative": ("benign", 0.27),
    }
    monkeypatch.setattr(
        error_analysis,
        "predict_one",
        lambda pipeline, text: predictions[text],
    )

    grouped = error_analysis.collect_misclassifications(
        Pipeline([]),
        examples,
        "held_out",
    )

    assert set(grouped) == {"benign_hard", "direct_injection"}
    records = grouped["benign_hard"] + grouped["direct_injection"]
    assert [record["text"] for record in records] == ["false positive", "false negative"]
    assert all(record["evaluation_set"] == "held_out" for record in records)
    assert records[0] == {
        "text": "false positive",
        "true_label": "benign",
        "predicted_label": "injection",
        "score": 0.81,
        "category": "benign_hard",
        "evaluation_set": "held_out",
    }


def test_main_writes_grouped_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "reports" / "errors.json"
    report: error_analysis.ErrorGroups = {
        "benign_hard": [
            {
                "text": "quoted fixture",
                "true_label": "benign",
                "predicted_label": "injection",
                "score": 0.73,
                "category": "benign_hard",
                "evaluation_set": "adversarial_suite",
            }
        ]
    }
    monkeypatch.setattr(error_analysis, "run_error_analysis", lambda: report)

    assert error_analysis.main(["--output", str(output_path)]) == 0

    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert "Wrote 1 misclassified examples" in capsys.readouterr().err
