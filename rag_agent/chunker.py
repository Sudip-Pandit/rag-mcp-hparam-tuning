"""1.1 Chunking Is Architecture, Not Preprocessing.

Token-aware recursive chunking with overlap, plus an indexing-cost estimator.
Falls back to a simple char-based splitter if langchain/tiktoken are absent.
"""

from typing import Callable, Dict, List
import re

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # tiktoken not installed
    _ENC = None

    def _count_tokens(text: str) -> int:
        # ~4 chars per token heuristic.
        return max(1, len(text) // 4)


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_with_separators(text: str, separators: List[str]) -> List[str]:
    """Recursive split that respects natural document structure."""
    if not separators:
        return [text]
    sep = separators[0]
    if sep == "":
        return list(text)
    parts = text.split(sep)
    # Re-attach separator so we don't lose it.
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def chunk_text(
    text: str,
    chunk_size: int = 512,      # tokens, NOT characters
    chunk_overlap: int = 64,    # ~12.5% overlap: sweet spot
) -> List[str]:
    """Split text into ~chunk_size token windows with chunk_overlap token overlap.

    Anti-pattern guard: refuses overlap above 25% of chunk_size.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if chunk_overlap > 0.25 * chunk_size:
        raise ValueError(
            f"chunk_overlap {chunk_overlap} exceeds 25% of chunk_size {chunk_size}. "
            "This doubles indexing cost for marginal recall gain (see article 1.1)."
        )

    # Tokenize into words then re-pack to token budgets (provider-agnostic).
    words = re.split(r"(\s+)", text)
    words = [w for w in words if w.strip() != ""]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for word in words:
        wt = _count_tokens(word + " ")
        if current_tokens + wt > chunk_size and current:
            chunks.append(" ".join(current).strip())
            # Build overlap window from the tail of the current chunk.
            overlap_words: List[str] = []
            ot = 0
            for w in reversed(current):
                t = _count_tokens(w + " ")
                if ot + t > chunk_overlap:
                    break
                overlap_words.insert(0, w)
                ot += t
            current = overlap_words
            current_tokens = ot
        current.append(word)
        current_tokens += wt

    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def estimate_indexing_cost(
    docs: List[str],
    chunk_size: int,
    chunk_overlap: int,
    cost_per_1k_tokens: float = 0.0001,
) -> Dict[str, float]:
    """Estimate embedding/indexing cost, including overlap waste."""
    total_input_tokens = sum(_count_tokens(d) for d in docs)
    overlap_multiplier = (
        chunk_size / (chunk_size - chunk_overlap) if chunk_overlap > 0 else 1.0
    )
    effective_tokens = total_input_tokens * overlap_multiplier
    return {
        "input_tokens": total_input_tokens,
        "effective_tokens": round(effective_tokens, 1),
        "overlap_multiplier": round(overlap_multiplier, 3),
        "cost_usd": round((effective_tokens / 1000) * cost_per_1k_tokens, 6),
        "waste_pct": round((overlap_multiplier - 1) * 100, 1),
    }


if __name__ == "__main__":
    sample = (
        "Chunking sounds boring. It is one of the most consequential decisions in a "
        "RAG pipeline. chunk_size decides how much information lives inside each "
        "segment. chunk_overlap decides how much context survives the boundaries "
        "between them. " * 10
    )
    out = chunk_text(sample, chunk_size=64, chunk_overlap=8)
    print(f"chunks: {len(out)}")
    print("first chunk:", out[0][:120], "...")
    print("cost est:", estimate_indexing_cost([sample], 512, 64))
