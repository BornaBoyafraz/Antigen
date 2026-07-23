import pytest

from benchmarks.latency import summarize_latencies


def test_summarize_latencies_reports_percentiles_in_milliseconds():
    summary = summarize_latencies([0.001] * 100)
    assert summary == {
        "p50_ms": pytest.approx(1.0),
        "p95_ms": pytest.approx(1.0),
        "p99_ms": pytest.approx(1.0),
    }


def test_summarize_latencies_requires_two_samples():
    with pytest.raises(ValueError, match="at least two"):
        summarize_latencies([0.001])
