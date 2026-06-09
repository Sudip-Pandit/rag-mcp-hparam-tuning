"""Cost evaluation helpers (Part 3.1)."""

from statistics import mean
from typing import Dict, List


def evaluate_cost(per_query_costs: List[float]) -> Dict[str, float]:
    if not per_query_costs:
        return {"avg_cost": 0.0, "total_cost": 0.0, "queries": 0}
    return {
        "queries": len(per_query_costs),
        "avg_cost": round(mean(per_query_costs), 6),
        "total_cost": round(sum(per_query_costs), 6),
        "monthly_at_100k_per_day": round(mean(per_query_costs) * 100_000 * 30, 2),
    }


if __name__ == "__main__":
    print(evaluate_cost([0.0021, 0.0019, 0.0025, 0.0030]))
