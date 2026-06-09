"""1.3 top_k, Similarity Metric, and Similarity Threshold.

Provides an in-memory vector store (works offline) plus the threshold-filtered
retrieval logic from the article. The same RetrievalConfig + retrieve_chunks
contract works against Pinecone if you swap the index object.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


class InsufficientContextError(Exception):
    """Raised when no retrieved chunk clears similarity_threshold."""


@dataclass
class RetrievalConfig:
    top_k: int = 8
    similarity_metric: str = "cosine"          # cosine | dot | l2
    similarity_threshold: float = 0.75
    namespace: Optional[str] = None
    filter_metadata: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Similarity helpers
# --------------------------------------------------------------------------- #
def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _l2_sim(a: List[float], b: List[float]) -> float:
    dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return 1.0 / (1.0 + dist)  # map distance to (0,1] similarity


_METRICS = {"cosine": _cosine, "dot": _dot, "l2": _l2_sim}


# --------------------------------------------------------------------------- #
# In-memory vector store (drop-in stand-in for Pinecone in demos)
# --------------------------------------------------------------------------- #
class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: List[Dict] = []  # {id, vector, metadata}

    def upsert(self, vectors: List[Dict]) -> None:
        self._items.extend(vectors)

    def query(
        self,
        vector: List[float],
        top_k: int,
        metric: str = "cosine",
        filter: Optional[dict] = None,
    ) -> List[Dict]:
        sim = _METRICS[metric]
        scored = []
        for item in self._items:
            if filter and not all(
                item["metadata"].get(k) == v for k, v in filter.items()
            ):
                continue
            scored.append(
                {
                    "id": item["id"],
                    "score": sim(vector, item["vector"]),
                    "metadata": item["metadata"],
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def retrieve_chunks(
    query_embedding: List[float],
    config: RetrievalConfig,
    index,
) -> List[dict]:
    """Retrieve with threshold filtering to avoid garbage context."""
    matches = index.query(
        vector=query_embedding,
        top_k=config.top_k,
        metric=config.similarity_metric,
        filter=config.filter_metadata,
    )

    filtered = [m for m in matches if m["score"] >= config.similarity_threshold]

    if not filtered:
        best = matches[0]["score"] if matches else 0.0
        raise InsufficientContextError(
            f"No chunks above threshold {config.similarity_threshold}. "
            f"Max score: {best:.3f}"
        )

    return [
        {"text": m["metadata"]["text"], "score": m["score"], "id": m["id"]}
        for m in filtered
    ]


if __name__ == "__main__":
    from embedder import EmbeddingConfig

    emb = EmbeddingConfig()
    store = InMemoryVectorStore()
    docs = [
        "Set temperature to zero for factual question answering.",
        "Chunk size determines retrieval precision in RAG.",
        "The capital of France is Paris.",
    ]
    for i, d in enumerate(docs):
        store.upsert([{"id": f"d{i}", "vector": emb.embed([d])[0],
                       "metadata": {"text": d}}])

    q = emb.embed(["how to reduce hallucination with temperature"])[0]
    cfg = RetrievalConfig(top_k=3, similarity_threshold=0.0)
    for r in retrieve_chunks(q, cfg, store):
        print(round(r["score"], 3), "->", r["text"])
