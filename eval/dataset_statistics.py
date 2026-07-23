"""Summarize dataset size, composition, and text lengths as plain tables.

Keeping the calculations in a standalone report makes corpus changes easy to
inspect before model evaluation and gives documentation claims a reproducible
source of row counts and class balance.
"""
from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

from model import Example, load_dataset


class CountRow(TypedDict):
    name: str
    rows: int
    share: float


@dataclass(frozen=True)
class DatasetStatistics:
    total_rows: int
    label_counts: list[CountRow]
    category_counts: list[CountRow]
    mean_text_length: float
    median_text_length: float


def _count_rows(values: Sequence[str], total: int) -> list[CountRow]:
    counts = Counter(values)
    return [
        {
            "name": name,
            "rows": count,
            "share": count / total,
        }
        for name, count in sorted(counts.items())
    ]


def calculate_dataset_statistics(examples: Sequence[Example]) -> DatasetStatistics:
    """Calculate counts, character lengths, and label/category shares."""
    if not examples:
        raise ValueError("dataset must contain at least one example")

    total = len(examples)
    text_lengths = [len(example.text) for example in examples]
    return DatasetStatistics(
        total_rows=total,
        label_counts=_count_rows([example.label for example in examples], total),
        category_counts=_count_rows([example.category for example in examples], total),
        mean_text_length=statistics.fmean(text_lengths),
        median_text_length=float(statistics.median(text_lengths)),
    )


def _format_count_table(title: str, first_column: str, rows: list[CountRow]) -> list[str]:
    lines = [
        title,
        f"{first_column:<28} {'rows':>8} {'share':>10}",
        f"{'-' * 28} {'-' * 8} {'-' * 10}",
    ]
    lines.extend(
        f"{row['name']:<28} {row['rows']:>8} {row['share']:>9.1%}" for row in rows
    )
    return lines


def format_report(report: DatasetStatistics) -> str:
    """Render dataset statistics in compact terminal-friendly tables."""
    lines = [
        "Dataset summary",
        f"{'measurement':<32} {'value':>12}",
        f"{'-' * 32} {'-' * 12}",
        f"{'total rows':<32} {report.total_rows:>12}",
        f"{'mean text length (characters)':<32} {report.mean_text_length:>12.2f}",
        f"{'median text length (characters)':<32} {report.median_text_length:>12.2f}",
        "",
    ]
    lines.extend(_format_count_table("Class balance", "label", report.label_counts))
    lines.append("")
    lines.extend(_format_count_table("Rows by category", "category", report.category_counts))
    return "\n".join(lines)


def main() -> int:
    """Load the generated corpus and print its current statistics."""
    print(format_report(calculate_dataset_statistics(load_dataset())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
