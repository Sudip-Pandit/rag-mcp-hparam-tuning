"""Latency evaluation helpers (Part 3.3)."""

from statistics import mean
from typing import Dict, List


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def evaluate_latency(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": round(mean(latencies_ms), 1),
        "p50_ms": round(_pct(latencies_ms, 0.50), 1),
        "p95_ms": round(_pct(latencies_ms, 0.95), 1),
        "p99_ms": round(_pct(latencies_ms, 0.99), 1),
    }


if __name__ == "__main__":
    print(evaluate_latency([12, 15, 20, 50, 9, 11, 300, 14]))
