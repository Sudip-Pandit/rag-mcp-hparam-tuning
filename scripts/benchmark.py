"""scripts/benchmark.py — Reproduce the article's two case studies.

Example 1: HR Policy Bot (RAG) before vs after tuning.
Example 2: Support Ticket Agent (MCP) before vs after tuning.
Runs offline. Run from the repo root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.config import RAGConfig
from rag_agent.pipeline import RAGPipeline
from mcp_agent.agent import default_support_agent


HR_DOCS = [
    "Current remote work policy: employees may work remote up to 3 days per week. "
    "status: current",
    "Old remote work policy from 18 months ago: no remote work allowed. "
    "status: deprecated",
    "Expense reimbursement is processed within 14 business days. status: current",
]


def hr_policy_bot_demo() -> None:
    print("=" * 60)
    print("Example 1: HR Policy Bot (RAG)")
    print("=" * 60)

    before = RAGConfig(chunk_size=1000, chunk_overlap=0, top_k=10,
                       similarity_threshold=0.0, temperature=0.7,
                       use_reranker=False, embedding_dim=256)
    after = RAGConfig(chunk_size=128, chunk_overlap=24, top_k=5,
                      similarity_threshold=0.0, temperature=0.0,
                      use_reranker=True, reranker_top_n=3, embedding_dim=256)

    q = "How many days can I work remotely?"
    for label, cfg in [("BEFORE", before), ("AFTER", after)]:
        pipe = RAGPipeline(cfg)
        # 'after' filters to current policy via metadata at query time.
        pipe.index_documents(HR_DOCS)
        if label == "AFTER":
            pipe.retrieval_cfg.filter_metadata = {"status": "current"}
        out = pipe.query(q)
        print(f"\n[{label}] temp={cfg.temperature} reranker={cfg.use_reranker}")
        print("  answer:", out["answer"])
        print("  latency_ms:", out["metrics"]["total_latency_ms"])


def support_agent_demo() -> None:
    print("\n" + "=" * 60)
    print("Example 2: Support Ticket Agent (MCP)")
    print("=" * 60)

    # AFTER tuning: destructive tools gated, temperature 0.0 implicit in routing.
    agent = default_support_agent(confirm_fn=None)  # no auto-approval

    cases = [
        "please add a comment that we are investigating the login issue",
        "close ticket T-100",     # destructive -> requires approval -> denied
        "delete ticket T-100",    # destructive -> denied
    ]
    for msg in cases:
        r = agent.run(msg)
        print(f"\nUSER: {msg}")
        print(f"AGENT: {r.answer}")
        print(f"  reason={r.stopped_reason} steps={r.steps} tokens={r.tokens_used}")

    print("\nWith human approval handler (confirm_fn returns True):")
    approving_agent = default_support_agent(confirm_fn=lambda name: True)
    r = approving_agent.run("close ticket T-100")
    print(f"AGENT: {r.answer} | reason={r.stopped_reason}")


if __name__ == "__main__":
    hr_policy_bot_demo()
    support_agent_demo()
