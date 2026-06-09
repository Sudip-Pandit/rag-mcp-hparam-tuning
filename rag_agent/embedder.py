"""1.2 Embedding Model and Dimension.

Centralizes embedding hyperparameters. Uses OpenAI when a client/key is
available, otherwise a deterministic local hashing embedder so the whole
pipeline runs offline for demos and tests.
"""

from dataclasses import dataclass
from typing import List, Optional
import hashlib
import math


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    normalize: bool = True
    batch_size: int = 512
    reduce_to_dim: Optional[int] = None

    def embed(self, texts: List[str], client=None) -> List[List[float]]:
        if client is not None:
            return self._embed_openai(texts, client)
        return [self._embed_local(t) for t in texts]

    # ------------------------------------------------------------------ #
    def _embed_openai(self, texts: List[str], client) -> List[List[float]]:
        resp = client.embeddings.create(
            input=texts, model=self.model, dimensions=self.dimensions
        )
        vecs = [r.embedding for r in resp.data]
        if self.normalize:
            vecs = [self._l2(v) for v in vecs]
        return vecs

    def _embed_local(self, text: str) -> List[float]:
        """Deterministic bag-of-hashed-tokens embedding (no deps)."""
        dim = self.reduce_to_dim or min(self.dimensions, 256)
        vec = [0.0] * dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        if self.normalize:
            vec = self._l2(vec)
        return vec

    @staticmethod
    def _l2(v: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


# Presets referenced in the article.
config_cheap = EmbeddingConfig(model="text-embedding-3-small", dimensions=256)
config_quality = EmbeddingConfig(model="text-embedding-3-large", dimensions=3072)


if __name__ == "__main__":
    cfg = EmbeddingConfig()
    v = cfg.embed(["temperature controls hallucination", "chunk size matters"])
    print(f"vectors: {len(v)}, dim: {len(v[0])}")
    print("sample[:5]:", [round(x, 4) for x in v[0][:5]])
