"""scripts/run_sweep.py — Hyperparameter sweep (Part 4.3).

Randomly samples up to max_experiments configs from SWEEP_GRID, evaluates each
on the synthetic QA set, and writes experiments/results/sweep_results.jsonl.
Runs fully offline.
"""

import itertools
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.config import RAGConfig
from rag_agent.pipeline import RAGPipeline
from evaluation.ragas_eval import evaluate_rag

SWEEP_GRID = {
    "chunk_size": [128, 256, 512],
    "chunk_overlap": [0, 16, 32],
    "top_k": [4, 8, 12],
    "use_reranker": [True, False],
    "temperature": [0.0, 0.1, 0.3],
    "reranker_top_n": [3, 4],
}


def _load_queries() -> list:
    p = Path("data/synthetic/qa_pairs.jsonl")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _load_docs() -> list:
    p = Path("data/synthetic/ground_truth.jsonl")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def run_sweep(max_experiments: int = 50, seed: int = 42) -> list:
    queries = _load_queries()
    doc_records = _load_docs()
    docs = [d["text"] for d in doc_records]

    out_path = Path("experiments/results")
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "sweep_results.jsonl"
    results_file.write_text("")  # reset

    keys, values = zip(*SWEEP_GRID.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    random.seed(seed)
    combos = random.sample(all_combos, min(max_experiments, len(all_combos)))

    results = []
    for i, combo in enumerate(combos):
        # overlap must stay <=25% of chunk_size (chunker enforces this).
        if combo["chunk_overlap"] > 0.25 * combo["chunk_size"]:
            combo = {**combo, "chunk_overlap": int(0.2 * combo["chunk_size"])}

        cfg = RAGConfig(
            embedding_dim=256,
            similarity_threshold=0.0,  # local embeddings -> demo-safe
            **combo,
        )
        pipe = RAGPipeline(cfg)
        pipe.index_documents(docs)

        eval_rows, costs, lats = [], [], []
        for q in queries:
            out = pipe.query(q["question"])
            retrieved_ids = [c["id"].split("_")[0].replace("doc", "d")
                             for c in out["chunks"]]
            eval_rows.append({
                "retrieved_ids": retrieved_ids,
                "relevant_ids": q["relevant_docs"],
                "answer": out["answer"],
                "context": [c["text"] for c in out["chunks"]],
            })
            costs.append(out["metrics"].get("cost", {}).get("total", 0.0))
            lats.append(out["metrics"]["total_latency_ms"])

        metrics = evaluate_rag(eval_rows)
        metrics["avg_cost"] = round(sum(costs) / len(costs), 6) if costs else 0.0
        lats_sorted = sorted(lats)
        p95_idx = int((len(lats_sorted) - 1) * 0.95)
        metrics["latency_p95_ms"] = round(lats_sorted[p95_idx], 1) if lats else 0.0

        rec = {
            "id": f"exp_{i:03d}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": {k: str(v) for k, v in combo.items()},
            "metrics": metrics,
        }
        results.append(rec)
        with open(results_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"exp_{i:03d} faith={metrics['faithfulness']:.3f} "
              f"hit={metrics['hit_rate']:.3f} cost=${metrics['avg_cost']:.5f}")

    best = max(results, key=lambda r: r["metrics"]["faithfulness"])
    print("\nBest config:", best["config"])
    print(f"  faithfulness: {best['metrics']['faithfulness']:.3f}")
    print(f"  avg cost/query: ${best['metrics']['avg_cost']:.5f}")
    print(f"  p95 latency: {best['metrics']['latency_p95_ms']:.0f}ms")
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_sweep(max_experiments=n)
