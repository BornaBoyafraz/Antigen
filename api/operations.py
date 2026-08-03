"""Operational middleware and in-memory request metrics for the API."""
from __future__ import annotations

import logging
import math
import os
import time
from collections import Counter, deque
from collections.abc import Callable
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

RATE_LIMIT_ENV_VAR = "ANTIGEN_RATE_LIMIT_PER_MINUTE"
DEFAULT_RATE_LIMIT_PER_MINUTE = 10_000
RATE_LIMIT_WINDOW_SECONDS = 60.0

request_logger = logging.getLogger("antigen.api.requests")


class RequestMetrics:
    """Thread-safe process-local counters and completed-request latencies."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        """Clear measurements and restart the uptime clock."""
        with self._lock:
            self._started_at = self._clock()
            self._total_requests = 0
            self._per_endpoint: Counter[str] = Counter()
            self._latencies_ms: list[float] = []

    def record_request(self, path: str) -> None:
        """Count a request when it enters the middleware stack."""
        with self._lock:
            self._total_requests += 1
            self._per_endpoint[path] += 1

    def record_latency(self, duration_ms: float) -> None:
        """Record the duration of a completed request."""
        with self._lock:
            self._latencies_ms.append(duration_ms)

    def snapshot(self) -> dict[str, object]:
        """Return a consistent JSON-ready view of the current measurements."""
        with self._lock:
            latencies = sorted(self._latencies_ms)
            average_ms = sum(latencies) / len(latencies) if latencies else 0.0
            p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
            p95_ms = latencies[p95_index] if latencies else 0.0
            return {
                "total_requests": self._total_requests,
                "per_endpoint": dict(sorted(self._per_endpoint.items())),
                "latency_ms": {
                    "avg": round(average_ms, 3),
                    "p95": round(p95_ms, 3),
                },
                "uptime_seconds": round(max(0.0, self._clock() - self._started_at), 3),
            }


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Measure and emit one structured standard-library log per HTTP request."""

    def __init__(self, app: ASGIApp, metrics: RequestMetrics) -> None:
        super().__init__(app)
        self._metrics = metrics

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        started_at = time.perf_counter()
        status = 500
        self._metrics.record_request(path)

        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            self._metrics.record_latency(duration_ms)
            request_logger.info(
                "request_completed",
                extra={
                    "method": request.method,
                    "path": path,
                    "status": status,
                    "duration_ms": round(duration_ms, 3),
                },
            )


def _configured_rate_limit() -> int:
    raw_limit = os.getenv(RATE_LIMIT_ENV_VAR)
    if raw_limit is None:
        return DEFAULT_RATE_LIMIT_PER_MINUTE

    try:
        limit = int(raw_limit)
    except ValueError as error:
        raise ValueError(f"{RATE_LIMIT_ENV_VAR} must be a positive integer") from error
    if limit <= 0:
        raise ValueError(f"{RATE_LIMIT_ENV_VAR} must be a positive integer")
    return limit


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply a process-local sliding-window request limit per client IP."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(app)
        self._requests_per_minute = (
            _configured_rate_limit() if requests_per_minute is None else requests_per_minute
        )
        if self._requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be a positive integer")
        self._clock = clock
        self._lock = Lock()
        self._request_times: dict[str, deque[float]] = {}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host if request.client is not None else "unknown"
        now = self._clock()

        with self._lock:
            request_times = self._request_times.setdefault(client_ip, deque())
            window_start = now - RATE_LIMIT_WINDOW_SECONDS
            while request_times and request_times[0] <= window_start:
                request_times.popleft()

            if len(request_times) >= self._requests_per_minute:
                retry_after = max(
                    1,
                    math.ceil(request_times[0] + RATE_LIMIT_WINDOW_SECONDS - now),
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
            request_times.append(now)

        return await call_next(request)
