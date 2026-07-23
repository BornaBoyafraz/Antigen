"""Tests for dataset count, balance, and text-length summaries."""
from __future__ import annotations

import pytest

from eval.dataset_statistics import calculate_dataset_statistics, format_report
from model import Example


def test_dataset_statistics_calculates_counts_lengths_and_balance() -> None:
    examples = [
        Example(text="a", label="benign", category="ordinary"),
        Example(text="abc", label="benign", category="ordinary"),
        Example(text="abcde", label="benign", category="tricky"),
        Example(text="abcdefg", label="injection", category="direct"),
    ]

    report = calculate_dataset_statistics(examples)

    assert report.total_rows == 4
    assert report.mean_text_length == 4.0
    assert report.median_text_length == 4.0
    assert report.label_counts == [
        {"name": "benign", "rows": 3, "share": 0.75},
        {"name": "injection", "rows": 1, "share": 0.25},
    ]
    assert report.category_counts == [
        {"name": "direct", "rows": 1, "share": 0.25},
        {"name": "ordinary", "rows": 2, "share": 0.5},
        {"name": "tricky", "rows": 1, "share": 0.25},
    ]


def test_dataset_statistics_formats_tables() -> None:
    examples = [
        Example(text="safe", label="benign", category="ordinary"),
        Example(text="attack", label="injection", category="direct"),
    ]

    rendered = format_report(calculate_dataset_statistics(examples))

    assert "Class balance" in rendered
    assert "Rows by category" in rendered
    assert "mean text length (characters)" in rendered
    assert rendered.count("50.0%") == 4


def test_dataset_statistics_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="at least one example"):
        calculate_dataset_statistics([])
