"""Index the synthetic ground-truth docs into an in-memory store and run a query.

Demonstrates the full RAG pipeline offline. Run from the repo root after
generate_synthetic_data.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.config import RAGConfig
from rag_agent.pipeline import RAGPipeline


def main() -> None:
    gt_path = Path("data/synthetic/ground_truth.jsonl")
    if not gt_path.exists():
        print("Run scripts/generate_synthetic_data.py first.")
        return

    docs = [json.loads(l)["text"] for l in gt_path.read_text().splitlines() if l.strip()]

    cfg = RAGConfig(
        chunk_size=128,
        chunk_overlap=16,
        embedding_dim=256,
        top_k=5,
        similarity_threshold=0.0,  # local hashing embeddings score low; demo-safe
        use_reranker=True,
        reranker_top_n=3,
        temperature=0.0,
    )
    pipe = RAGPipeline(cfg)
    n = pipe.index_documents(docs)
    print(f"Indexed {n} chunks.\n")

    for q in ["What temperature should I use for factual QA?",
              "How do I stop confident off-topic answers?"]:
        out = pipe.query(q)
        print("Q:", q)
        print("A:", out["answer"])
        print("latency_ms:", out["metrics"]["total_latency_ms"],
              "| cost:", out["metrics"].get("cost", {}).get("total"), "\n")


if __name__ == "__main__":
    main()
