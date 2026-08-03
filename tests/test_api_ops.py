"""Tests for API request controls, telemetry, and schema metadata."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import operations
from api.app import app


@pytest.fixture
def ops_client() -> Iterator[TestClient]:
    """Yield an API client with request metrics isolated from other tests."""
    metrics: operations.RequestMetrics = app.state.request_metrics
    metrics.reset()
    with TestClient(app) as client:
        yield client
    metrics.reset()


def test_request_logging_includes_structured_fields(
    ops_client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.clear()
    caplog.set_level(logging.INFO, logger="antigen.api.requests")

    response = ops_client.get("/api/health")

    assert response.status_code == 200
    records = [record for record in caplog.records if record.name == "antigen.api.requests"]
    assert len(records) == 1
    record = records[0]
    assert record.__dict__["method"] == "GET"
    assert record.__dict__["path"] == "/api/health"
    assert record.__dict__["status"] == 200
    duration_ms = record.__dict__["duration_ms"]
    assert isinstance(duration_ms, (int, float))
    assert duration_ms >= 0.0


def test_rate_limiter_reads_environment_and_tracks_each_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTIGEN_RATE_LIMIT_PER_MINUTE", "2")
    limited_app = FastAPI()
    limited_app.add_middleware(operations.RateLimitMiddleware)

    @limited_app.get("/probe")
    def probe() -> dict[str, str]:
        return {"status": "ok"}

    first_ip = ("192.0.2.1", 50000)
    second_ip = ("198.51.100.2", 50000)
    with (
        TestClient(limited_app, client=first_ip) as first_client,
        TestClient(limited_app, client=second_ip) as second_client,
    ):
        assert first_client.get("/probe").status_code == 200
        assert first_client.get("/probe").status_code == 200

        blocked = first_client.get("/probe")
        assert blocked.status_code == 429
        assert blocked.json() == {"detail": "Rate limit exceeded"}

        assert second_client.get("/probe").status_code == 200


def test_metrics_report_counts_latencies_and_uptime(ops_client: TestClient) -> None:
    assert ops_client.get("/api/health").status_code == 200
    assert ops_client.get("/api/examples").status_code == 200

    response = ops_client.get("/api/metrics")

    assert response.status_code == 200
    metrics = response.json()
    assert metrics["total_requests"] >= 2
    assert metrics["per_endpoint"]["/api/health"] == 1
    assert metrics["per_endpoint"]["/api/examples"] == 1
    assert metrics["latency_ms"]["avg"] >= 0.0
    assert metrics["latency_ms"]["p95"] >= 0.0
    assert metrics["uptime_seconds"] >= 0.0


def test_api_operations_have_openapi_metadata_and_classify_example() -> None:
    schema = app.openapi()
    api_paths = {
        path: operations for path, operations in schema["paths"].items() if path.startswith("/api/")
    }
    assert api_paths
    for path_operations in api_paths.values():
        for operation in path_operations.values():
            assert operation["tags"]
            assert operation["summary"]

    classify = api_paths["/api/classify"]["post"]
    example = classify["responses"]["200"]["content"]["application/json"]["example"]
    assert example["label"] in {"benign", "injection"}
    assert 0.0 <= example["score"] <= 1.0
    assert isinstance(example["explanation"], dict)
