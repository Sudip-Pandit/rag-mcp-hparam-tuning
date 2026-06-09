"""1.4 Reranker Model and top_n.

Two-stage retrieval: cast wide with vector search, precision-cut with a
cross-encoder reranker. Uses sentence-transformers CrossEncoder or Cohere when
available; otherwise a lightweight lexical-overlap scorer so the pipeline runs
offline.
"""

from dataclasses import dataclass
from typing import List
import time


@dataclass
class RerankerConfig:
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 4
    score_threshold: float = 0.0
    use_cohere: bool = False
    cohere_model: str = "rerank-english-v3.0"

    def rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        t0 = time.time()
        if self.use_cohere:
            reranked = self._rerank_cohere(query, chunks)
        else:
            reranked = self._rerank_local(query, chunks)
        elapsed = time.time() - t0
        # Trim, then report.
        reranked = reranked[: self.top_n]
        print(f"[reranker] {len(chunks)} -> {len(reranked)} in {elapsed*1000:.0f}ms")
        return reranked

    # ------------------------------------------------------------------ #
    def _rerank_cohere(self, query: str, chunks: List[dict]) -> List[dict]:
        from cohere import Client  # imported lazily

        co = Client()
        docs = [c["text"] for c in chunks]
        results = co.rerank(
            query=query, documents=docs, model=self.cohere_model, top_n=self.top_n
        )
        return [
            {**chunks[r.index], "rerank_score": float(r.relevance_score)}
            for r in results.results
        ]

    def _rerank_local(self, query: str, chunks: List[dict]) -> List[dict]:
        """Try CrossEncoder; fall back to lexical overlap if unavailable."""
        try:
            from sentence_transformers import CrossEncoder

            ce = CrossEncoder(self.model)
            pairs = [(query, c["text"]) for c in chunks]
            scores = ce.predict(pairs)
            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            return [{**c, "rerank_score": float(s)} for c, s in ranked]
        except Exception:
            q_terms = set(query.lower().split())
            scored = []
            for c in chunks:
                terms = set(c["text"].lower().split())
                overlap = len(q_terms & terms) / (len(q_terms) or 1)
                scored.append((c, overlap))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [{**c, "rerank_score": round(s, 4)} for c, s in scored]


if __name__ == "__main__":
    chunks = [
        {"id": "1", "text": "temperature zero reduces hallucination", "score": 0.8},
        {"id": "2", "text": "paris is the capital of france", "score": 0.7},
        {"id": "3", "text": "lower temperature for factual answers", "score": 0.6},
    ]
    rc = RerankerConfig(top_n=2)
    for r in rc.rerank("how does temperature affect hallucination", chunks):
        print(round(r["rerank_score"], 3), r["text"])
