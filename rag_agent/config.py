"""Master RAG hyperparameter configuration (Part 4.3).

All tunable parameters live here so a sweep can vary any of them.
"""

from dataclasses import dataclass

from .query_rewriter import QueryRewriteStrategy


@dataclass
class RAGConfig:
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    # Retrieval
    top_k: int = 8
    similarity_metric: str = "cosine"
    similarity_threshold: float = 0.75
    # Reranker
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 4
    # Query rewriting
    query_rewrite_strategy: QueryRewriteStrategy = QueryRewriteStrategy.NONE
    query_rewrite_n_variants: int = 3
    query_rewrite_temperature: float = 0.3
    # Generation
    llm_model: str = "gpt-4o"
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 512
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
