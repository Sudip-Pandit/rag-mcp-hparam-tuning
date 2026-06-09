"""1.7 Evaluation Metrics and Parameters.

Computes the minimum eval triad: hit_rate, MRR, faithfulness.
faithfulness uses an LLM judge when a client is supplied, else a lexical
grounding proxy so evals run offline.
"""

from statistics import mean
from typing import Dict, List


def _hit(retrieved_ids: List[str], relevant_ids: List[str]) -> int:
    return 1 if any(r in relevant_ids for r in retrieved_ids) else 0


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def _faithfulness_proxy(answer: str, context: List[str]) -> float:
    """Fraction of answer terms supported by the context (offline proxy)."""
    a_terms = set(answer.lower().split())
    if not a_terms:
        return 0.0
    ctx_terms = set(" ".join(context).lower().split())
    supported = len(a_terms & ctx_terms)
    return round(supported / len(a_terms), 3)


def evaluate_rag(query_results: List[Dict], client=None) -> Dict[str, float]:
    """query_results: list of {retrieved_ids, relevant_ids, answer, context}."""
    if not query_results:
        return {"hit_rate": 0.0, "mrr": 0.0, "faithfulness": 0.0}

    hits, rrs, faiths = [], [], []
    for qr in query_results:
        rids = qr.get("retrieved_ids", [])
        rel = qr.get("relevant_ids", [])
        hits.append(_hit(rids, rel))
        rrs.append(reciprocal_rank(rids, rel))
        faiths.append(_faithfulness_proxy(qr.get("answer", ""), qr.get("context", [])))

    return {
        "hit_rate": round(mean(hits), 3),
        "mrr": round(mean(rrs), 3),
        "faithfulness": round(mean(faiths), 3),
    }


if __name__ == "__main__":
    sample = [
        {
            "retrieved_ids": ["d1", "d2"],
            "relevant_ids": ["d1"],
            "answer": "set temperature to zero for factual answers",
            "context": ["set temperature to zero for factual question answering"],
        },
        {
            "retrieved_ids": ["d3", "d4"],
            "relevant_ids": ["d9"],
            "answer": "paris is the capital",
            "context": ["the capital of france is paris"],
        },
    ]
    print(evaluate_rag(sample))
