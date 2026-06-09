"""Generate synthetic QA pairs and ground-truth docs for evaluation.

Writes data/synthetic/qa_pairs.jsonl and ground_truth.jsonl. Self-contained:
no API key needed. Run from the repo root.
"""

import json
from pathlib import Path

DOCS = [
    ("d0", "Set temperature to 0.0 for factual question answering. Higher temperature "
           "increases hallucination in grounded RAG tasks."),
    ("d1", "chunk_size determines retrieval precision and is locked at index time. "
           "A bad chunking decision is a permanent tax until you re-index."),
    ("d2", "similarity_threshold above 0.72 prevents confident answers on off-topic "
           "queries. Without it the LLM uses irrelevant context."),
    ("d3", "A reranker is a cross-encoder that scores query-chunk pairs. The two-stage "
           "pattern retrieves top_k then reranks to top_n for better precision."),
    ("d4", "top_k controls how many chunks you retrieve. top_k=20 triples latency and "
           "cost for marginal quality gain compared to top_k=8."),
    ("d5", "In MCP agents, set temperature 0.0 for tool calling to avoid tool name "
           "hallucination. Always define max_steps and an execution budget."),
    ("d6", "Destructive tools must be gated behind human confirmation. require "
           "human approval before close or delete operations."),
    ("d7", "memory_window_turns controls how many conversation turns are sent each "
           "call. Larger windows cost more; summarize evicted turns."),
]

QA_PAIRS = [
    ("What temperature should I use for factual QA?",
     "Use temperature 0.0 for factual question answering.", ["d0"]),
    ("Why does chunk size matter so much?",
     "chunk_size sets retrieval precision and is fixed at index time.", ["d1"]),
    ("How do I stop confident off-topic answers?",
     "Set a similarity_threshold above 0.72.", ["d2"]),
    ("What is a reranker?",
     "A cross-encoder that scores query-chunk pairs in a two-stage pipeline.", ["d3"]),
    ("Is top_k=20 a good default?",
     "No, top_k=20 triples cost and latency over top_k=8 for little gain.", ["d4"]),
    ("How do I prevent tool name hallucination in MCP?",
     "Set temperature 0.0 for tool calling and bound max_steps.", ["d5"]),
    ("How should destructive tools be handled?",
     "Gate them behind human confirmation.", ["d6"]),
    ("What does memory_window_turns control?",
     "How many conversation turns are included per agent call.", ["d7"]),
]


def main() -> None:
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ground_truth.jsonl", "w") as f:
        for did, text in DOCS:
            f.write(json.dumps({"id": did, "text": text}) + "\n")

    with open(out_dir / "qa_pairs.jsonl", "w") as f:
        for q, a, rel in QA_PAIRS:
            f.write(json.dumps({"question": q, "answer": a,
                                "relevant_docs": rel}) + "\n")

    print(f"Wrote {len(DOCS)} docs and {len(QA_PAIRS)} QA pairs to {out_dir}")


if __name__ == "__main__":
    main()
