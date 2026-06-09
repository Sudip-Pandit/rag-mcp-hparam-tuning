"""Sweep report generator (Part 4.3).

Reads experiments/results/sweep_results.jsonl and prints a ranked table plus the
best config by faithfulness, with cost and latency tie-breakers.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_results(path: str = "experiments/results/sweep_results.jsonl") -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def generate_report(results: List[Dict]) -> str:
    if not results:
        return "No sweep results found. Run scripts/run_sweep.py first."

    ranked = sorted(
        results,
        key=lambda r: (
            r["metrics"].get("faithfulness", 0),
            -r["metrics"].get("avg_cost", 1e9),
        ),
        reverse=True,
    )

    lines = ["Rank | exp_id   | faith | hit_rate | mrr   | cost     | p95_ms",
             "-" * 64]
    for i, r in enumerate(ranked[:15], 1):
        m = r["metrics"]
        lines.append(
            f"{i:>4} | {r['id']:<8} | "
            f"{m.get('faithfulness', 0):.3f} | "
            f"{m.get('hit_rate', 0):.3f}    | "
            f"{m.get('mrr', 0):.3f} | "
            f"${m.get('avg_cost', 0):.5f} | "
            f"{m.get('latency_p95_ms', 0):.0f}"
        )

    best = ranked[0]
    lines += [
        "",
        "BEST CONFIG (by faithfulness, cost tie-break):",
        json.dumps(best["config"], indent=2, default=str),
        f"faithfulness={best['metrics'].get('faithfulness', 0):.3f}  "
        f"avg_cost=${best['metrics'].get('avg_cost', 0):.5f}  "
        f"p95={best['metrics'].get('latency_p95_ms', 0):.0f}ms",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_report(load_results()))
