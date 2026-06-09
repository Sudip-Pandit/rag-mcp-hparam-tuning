"""1.5 Query Rewriting Parameters.

Supports NONE / HyDE / MULTI_QUERY / STEP_BACK / RAG_FUSION strategies.
Uses an OpenAI client when provided; otherwise returns deterministic
rule-based variants so the pipeline runs offline.
"""

from enum import Enum
from typing import List, Optional


class QueryRewriteStrategy(Enum):
    NONE = "none"
    HYDE = "hyde"
    MULTI_QUERY = "multi"
    STEP_BACK = "stepback"
    RAG_FUSION = "fusion"


def rewrite_query(
    query: str,
    strategy: QueryRewriteStrategy,
    client=None,
    n_variants: int = 3,
    rewrite_temperature: float = 0.3,
    model: str = "gpt-4o-mini",
) -> List[str]:
    if strategy == QueryRewriteStrategy.NONE:
        return [query]

    if client is None:
        return _offline_rewrite(query, strategy, n_variants)

    if strategy == QueryRewriteStrategy.HYDE:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Write a hypothetical document that would answer: '{query}'",
            }],
            temperature=rewrite_temperature, max_tokens=200,
        )
        return [resp.choices[0].message.content]

    if strategy in (QueryRewriteStrategy.MULTI_QUERY, QueryRewriteStrategy.RAG_FUSION):
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    f"Generate {n_variants} search queries for: '{query}'. "
                    "Output one per line, no numbers."
                ),
            }],
            temperature=rewrite_temperature, max_tokens=150,
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        variants = [q.strip() for q in lines if q.strip()][:n_variants]
        return [query] + variants

    if strategy == QueryRewriteStrategy.STEP_BACK:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Write a more general 'step-back' question for: '{query}'",
            }],
            temperature=rewrite_temperature, max_tokens=80,
        )
        return [query, resp.choices[0].message.content.strip()]

    return [query]


def _offline_rewrite(
    query: str, strategy: QueryRewriteStrategy, n_variants: int
) -> List[str]:
    base = query.rstrip("?")
    if strategy == QueryRewriteStrategy.HYDE:
        return [f"{base}. The answer is as follows:"]
    if strategy == QueryRewriteStrategy.STEP_BACK:
        return [query, f"What general concept underlies: {base}?"]
    # MULTI_QUERY / RAG_FUSION
    templates = [
        f"{base}",
        f"how to {base}",
        f"best practices for {base}",
        f"{base} explained",
    ]
    return [query] + templates[1 : n_variants + 1]


if __name__ == "__main__":
    for s in QueryRewriteStrategy:
        print(s.value, "->", rewrite_query("reduce RAG latency", s, n_variants=2))
