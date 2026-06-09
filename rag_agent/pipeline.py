"""End-to-end RAG pipeline (Part 4.3).

Wires chunking -> embedding -> retrieval -> rerank -> generation, with cost and
latency tracking. Runs fully offline (in-memory store + local embeddings) when
no API keys are supplied, and against OpenAI + Pinecone when they are.
"""

from typing import Any, Dict, List, Optional
import os
import time

from .config import RAGConfig
from .chunker import chunk_text, _count_tokens
from .embedder import EmbeddingConfig
from .reranker import RerankerConfig
from .retriever import (
    InMemoryVectorStore,
    RetrievalConfig,
    retrieve_chunks,
    InsufficientContextError,
)
from .query_rewriter import rewrite_query
from .generator import GenerationConfig, generate_rag_answer

from shared.cost_tracker import CostTracker
from shared.latency_tracker import LatencyTracker


def _maybe_openai():
    """Return an OpenAI client if the SDK and key are present, else None."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        return None


class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        index=None,
        openai_client: Optional[Any] = None,
    ):
        self.config = config
        self.openai = openai_client if openai_client is not None else _maybe_openai()
        self.index = index if index is not None else InMemoryVectorStore()

        self.embed_cfg = EmbeddingConfig(
            model=config.embedding_model, dimensions=config.embedding_dim
        )
        self.retrieval_cfg = RetrievalConfig(
            top_k=config.top_k,
            similarity_metric=config.similarity_metric,
            similarity_threshold=config.similarity_threshold,
        )
        self.reranker_cfg = (
            RerankerConfig(top_n=config.reranker_top_n, model=config.reranker_model)
            if config.use_reranker
            else None
        )
        self.gen_cfg = GenerationConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            model=config.llm_model,
        )
        self.cost = CostTracker()
        self.latency = LatencyTracker()

    # ------------------------------------------------------------------ #
    def index_documents(self, docs: List[str]) -> int:
        """Chunk + embed + upsert. Returns number of chunks indexed."""
        n = 0
        for doc_id, doc in enumerate(docs):
            chunks = chunk_text(doc, self.config.chunk_size, self.config.chunk_overlap)
            vectors = []
            for ci, ch in enumerate(chunks):
                vec = self.embed_cfg.embed([ch], client=self.openai)[0]
                vectors.append({
                    "id": f"doc{doc_id}_c{ci}",
                    "vector": vec,
                    "metadata": {"text": ch, "doc_id": doc_id, "status": "current"},
                })
            self.index.upsert(vectors)
            n += len(vectors)
        return n

    def query(self, user_query: str) -> Dict[str, Any]:
        t_start = time.time()
        metrics: Dict[str, Any] = {"query": user_query}

        # 1. Rewrite
        with self.latency.measure("query_rewrite"):
            queries = rewrite_query(
                user_query,
                self.config.query_rewrite_strategy,
                client=self.openai,
                n_variants=self.config.query_rewrite_n_variants,
                rewrite_temperature=self.config.query_rewrite_temperature,
            )

        # 2. Retrieve (dedup across rewritten queries)
        with self.latency.measure("retrieval"):
            all_chunks: List[dict] = []
            for q in queries:
                emb = self.embed_cfg.embed([q], client=self.openai)[0]
                try:
                    all_chunks.extend(retrieve_chunks(emb, self.retrieval_cfg, self.index))
                except InsufficientContextError:
                    continue
            seen = set()
            unique = [
                c for c in all_chunks if not (c["id"] in seen or seen.add(c["id"]))
            ]
        metrics["chunks_retrieved"] = len(unique)

        if not unique:
            metrics["total_latency_ms"] = (time.time() - t_start) * 1000
            return {"answer": "I don't have that information.",
                    "chunks": [], "metrics": metrics}

        # 3. Rerank
        with self.latency.measure("rerank"):
            if self.reranker_cfg:
                unique = self.reranker_cfg.rerank(user_query, unique)
        metrics["chunks_after_rerank"] = len(unique)

        # 4. Generate
        with self.latency.measure("generation"):
            context_texts = [c["text"] for c in unique]
            answer = generate_rag_answer(
                user_query, context_texts, self.gen_cfg, client=self.openai
            )

        # Cost accounting
        input_tokens = sum(_count_tokens(t) for t in context_texts)
        out_tokens = _count_tokens(answer)
        self.cost.record_query(
            embedding_tokens=sum(_count_tokens(q) for q in queries),
            input_tokens=input_tokens,
            output_tokens=out_tokens,
            gen_model=self.config.llm_model
            if self.config.llm_model in ("gpt-4o", "gpt-4o-mini")
            else "gpt-4o",
            reranker_calls=1 if self.reranker_cfg else 0,
        )

        total_ms = (time.time() - t_start) * 1000
        self.latency.record_total(total_ms)
        metrics["total_latency_ms"] = round(total_ms, 1)
        metrics["cost"] = self.cost.records[-1].as_dict()

        return {"answer": answer, "chunks": unique, "metrics": metrics}


if __name__ == "__main__":
    cfg = RAGConfig(similarity_threshold=0.0, use_reranker=True, embedding_dim=256)
    pipe = RAGPipeline(cfg)
    pipe.index_documents([
        "Set temperature to 0.0 for factual question answering to cut hallucination.",
        "chunk_size controls retrieval precision and is locked at index time.",
        "similarity_threshold above 0.72 prevents confident off-topic answers.",
    ])
    out = pipe.query("how do I stop my RAG system from hallucinating")
    print("ANSWER:", out["answer"])
    print("METRICS:", out["metrics"])
