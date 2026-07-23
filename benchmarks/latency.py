"""Measure local classification latency and batch throughput.

The benchmark mirrors the API response path by producing both a calibrated
classification and its explanation. Model training and warmup are excluded so
the reported timings describe steady-state inference rather than startup.
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence

from sklearn.pipeline import Pipeline

from explain import explain, fit_explainer
from model import Example, load_dataset, train


def summarize_latencies(samples_seconds: list[float]) -> dict[str, float]:
    """Return inclusive p50, p95, and p99 latencies in milliseconds."""
    if len(samples_seconds) < 2:
        raise ValueError("at least two latency samples are required")

    percentiles = statistics.quantiles(samples_seconds, n=100, method="inclusive")
    return {
        "p50_ms": statistics.median(samples_seconds) * 1000,
        "p95_ms": percentiles[94] * 1000,
        "p99_ms": percentiles[98] * 1000,
    }


def _classify(pipeline: Pipeline, explainer: Pipeline, text: str) -> None:
    explain(pipeline, explainer, text).to_dict()


def run_benchmark(
    examples: list[Example],
    runs: int,
    batch_size: int,
    batches: int,
    warmup: int,
) -> tuple[dict[str, float], float]:
    """Train once, then measure per-item latency and batched item throughput."""
    if runs < 2:
        raise ValueError("runs must be at least 2")
    if batch_size < 1 or batches < 1 or warmup < 0:
        raise ValueError("batch_size and batches must be positive; warmup cannot be negative")
    if not examples:
        raise ValueError("examples cannot be empty")

    pipeline = train(examples)
    explainer = fit_explainer(examples)

    for index in range(warmup):
        _classify(pipeline, explainer, examples[index % len(examples)].text)

    samples = []
    for index in range(runs):
        text = examples[index % len(examples)].text
        started = time.perf_counter()
        _classify(pipeline, explainer, text)
        samples.append(time.perf_counter() - started)

    total_items = batch_size * batches
    started = time.perf_counter()
    for index in range(total_items):
        _classify(pipeline, explainer, examples[index % len(examples)].text)
    batch_seconds = time.perf_counter() - started
    throughput = total_items / batch_seconds

    return summarize_latencies(samples), throughput


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=500, help="measured single classifications")
    parser.add_argument("--batch-size", type=int, default=50, help="items per logical batch")
    parser.add_argument("--batches", type=int, default=20, help="number of batches to measure")
    parser.add_argument("--warmup", type=int, default=20, help="unmeasured warmup classifications")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary, throughput = run_benchmark(
        load_dataset(),
        runs=args.runs,
        batch_size=args.batch_size,
        batches=args.batches,
        warmup=args.warmup,
    )

    rows = [
        ("single p50", summary["p50_ms"], "ms"),
        ("single p95", summary["p95_ms"], "ms"),
        ("single p99", summary["p99_ms"], "ms"),
        (f"batch throughput ({args.batch_size}/batch)", throughput, "items/s"),
    ]
    print(
        f"Measured {args.runs} single classifications and "
        f"{args.batch_size * args.batches} batched items "
        f"after {args.warmup} warmups; training excluded.\n"
    )
    print(f"{'measurement':<34} {'value':>10}  unit")
    print(f"{'-' * 34} {'-' * 10}  {'-' * 7}")
    for name, value, unit in rows:
        print(f"{name:<34} {value:>10.3f}  {unit}")


if __name__ == "__main__":
    main()
