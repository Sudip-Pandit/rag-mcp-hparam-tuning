"""Cost tracking for RAG and MCP pipelines.

Implements the cost model from Part 3.1 of the article:

    cost_per_query =
        (embedding_tokens   x embedding_cost)
      + (top_k x avg_chunk_tokens x input_cost)
      + (reranker_calls     x reranker_cost)
      + (query_rewrite_tokens x rewrite_model_cost)
      + (output_tokens      x output_cost)
"""

from dataclasses import dataclass, field
from typing import Dict, List


# Prices are USD per 1K tokens unless noted. Update as provider pricing changes.
PRICING = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-3-large": 0.00013,
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    # Cohere rerank is priced per search (1K docs/search). ~$2 / 1M queries.
    "cohere-rerank": 0.000002,  # per query
}


@dataclass
class CostBreakdown:
    embedding_cost: float = 0.0
    input_cost: float = 0.0
    reranker_cost: float = 0.0
    rewrite_cost: float = 0.0
    output_cost: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.embedding_cost
            + self.input_cost
            + self.reranker_cost
            + self.rewrite_cost
            + self.output_cost
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "embedding_cost": round(self.embedding_cost, 6),
            "input_cost": round(self.input_cost, 6),
            "reranker_cost": round(self.reranker_cost, 6),
            "rewrite_cost": round(self.rewrite_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total": round(self.total, 6),
        }


class CostTracker:
    """Accumulates per-query cost and reports aggregate stats."""

    def __init__(self) -> None:
        self.records: List[CostBreakdown] = []

    def record_query(
        self,
        *,
        embedding_tokens: int = 0,
        embedding_model: str = "text-embedding-3-small",
        input_tokens: int = 0,
        output_tokens: int = 0,
        rewrite_tokens: int = 0,
        gen_model: str = "gpt-4o",
        rewrite_model: str = "gpt-4o-mini",
        reranker_calls: int = 0,
    ) -> CostBreakdown:
        gen_price = PRICING[gen_model]
        rewrite_price = PRICING[rewrite_model]

        cb = CostBreakdown(
            embedding_cost=(embedding_tokens / 1000) * PRICING[embedding_model],
            input_cost=(input_tokens / 1000) * gen_price["input"],
            reranker_cost=reranker_calls * PRICING["cohere-rerank"],
            rewrite_cost=(rewrite_tokens / 1000) * rewrite_price["input"],
            output_cost=(output_tokens / 1000) * gen_price["output"],
        )
        self.records.append(cb)
        return cb

    def summary(self) -> Dict[str, float]:
        if not self.records:
            return {"queries": 0, "avg_cost": 0.0, "total_cost": 0.0}
        total = sum(r.total for r in self.records)
        return {
            "queries": len(self.records),
            "avg_cost": round(total / len(self.records), 6),
            "total_cost": round(total, 6),
        }


if __name__ == "__main__":
    t = CostTracker()
    # Simulate one tuned query: top_k=8, 512-token chunks, 200-token answer.
    t.record_query(
        embedding_tokens=12,
        input_tokens=8 * 512,
        output_tokens=200,
        reranker_calls=1,
    )
    print("cost breakdown:", t.records[0].as_dict())
    print("summary:", t.summary())
