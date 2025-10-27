"""Prometheus instrumentation helpers for the VCT services."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

API_REQUEST_COUNTER = Counter(
    "vct_api_requests_total",
    "Total number of HTTP requests handled by the VCT API",
    ("endpoint", "method", "status"),
)

COMMAND_LATENCY = Histogram(
    "vct_command_latency_seconds",
    "Time spent handling RoboDog commands",
    ("endpoint",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

COMMAND_COUNTER = Counter(
    "vct_commands_total",
    "Commands processed by RoboDogBrain",
    ("source",),
)

REWARD_COUNTER = Counter(
    "vct_rewards_total",
    "Reward actuator outcomes",
    ("action", "outcome"),
)


def record_api_request(endpoint: str, method: str, status_code: int) -> None:
    """Increment API request counter with labels for downstream analysis."""

    API_REQUEST_COUNTER.labels(
        endpoint=endpoint, method=method.upper(), status=str(status_code)
    ).inc()


def observe_command_latency(endpoint: str, duration_s: float) -> None:
    """Observe command processing latency for the given endpoint."""

    COMMAND_LATENCY.labels(endpoint=endpoint).observe(duration_s)


def record_command(source: str) -> None:
    """Track command processing by its source (e.g. API or CLI)."""

    COMMAND_COUNTER.labels(source=source).inc()


def record_reward(action: str, rewarded: bool) -> None:
    """Track reward actuator outcomes per action."""

    outcome = "rewarded" if rewarded else "skipped"
    REWARD_COUNTER.labels(action=action or "UNKNOWN", outcome=outcome).inc()


__all__ = [
    "API_REQUEST_COUNTER",
    "COMMAND_COUNTER",
    "COMMAND_LATENCY",
    "REWARD_COUNTER",
    "observe_command_latency",
    "record_api_request",
    "record_command",
    "record_reward",
]
