"""Latency tracking with percentile reporting (Part 3.3).

Tracks per-stage timings and computes P50/P95/P99 across queries.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List
import time


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


@dataclass
class LatencyTracker:
    stage_times: Dict[str, List[float]] = field(default_factory=dict)
    totals: List[float] = field(default_factory=list)

    @contextmanager
    def measure(self, stage: str):
        t0 = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - t0) * 1000
            self.stage_times.setdefault(stage, []).append(elapsed_ms)

    def record_total(self, total_ms: float) -> None:
        self.totals.append(total_ms)

    def report(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for stage, vals in self.stage_times.items():
            out[stage] = {
                "mean_ms": round(mean(vals), 1),
                "p50_ms": round(_percentile(vals, 0.50), 1),
                "p95_ms": round(_percentile(vals, 0.95), 1),
            }
        if self.totals:
            out["total"] = {
                "mean_ms": round(mean(self.totals), 1),
                "p50_ms": round(_percentile(self.totals, 0.50), 1),
                "p95_ms": round(_percentile(self.totals, 0.95), 1),
                "p99_ms": round(_percentile(self.totals, 0.99), 1),
            }
        return out


if __name__ == "__main__":
    lt = LatencyTracker()
    for i in range(20):
        with lt.measure("retrieval"):
            time.sleep(0.002 + i * 0.0001)
        with lt.measure("generation"):
            time.sleep(0.01)
        lt.record_total(15 + i)
    import json
    print(json.dumps(lt.report(), indent=2))
