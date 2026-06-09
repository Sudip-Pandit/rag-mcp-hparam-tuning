"""1.6 Generation Parameters: temperature, top_p, max_tokens, penalties.

Generates the grounded answer. Uses OpenAI when a client is provided; otherwise
an extractive fallback that quotes the most relevant context sentence so the
pipeline produces a deterministic, grounded answer offline.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GenerationConfig:
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 512
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    model: str = "gpt-4o"
    system_prompt: str = field(default_factory=lambda: (
        "You are a precise Q&A assistant. Answer ONLY from the provided context. "
        "If the answer is not in the context, say 'I don't have that information.' "
        "Never fabricate facts. Be concise."
    ))


def generate_rag_answer(
    query: str,
    context_chunks: List[str],
    config: GenerationConfig,
    client=None,
) -> str:
    if client is None:
        return _extractive_answer(query, context_chunks)

    context = "\n\n---\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        stream=config.stream,
    )
    if config.stream:
        return "".join(c.choices[0].delta.content or "" for c in response)
    return response.choices[0].message.content


def _extractive_answer(query: str, context_chunks: List[str]) -> str:
    """Offline grounding: pick the context sentence most overlapping the query."""
    if not context_chunks:
        return "I don't have that information."
    q_terms = set(query.lower().split())
    best, best_score = None, -1.0
    for chunk in context_chunks:
        for sent in chunk.replace("\n", " ").split(". "):
            terms = set(sent.lower().split())
            score = len(q_terms & terms)
            if score > best_score:
                best, best_score = sent.strip(), score
    if best_score <= 0:
        # No lexical overlap: fall back to the highest-ranked chunk's first
        # sentence rather than claiming no information (chunks already cleared
        # the similarity_threshold upstream).
        first = context_chunks[0].replace("\n", " ").split(". ")[0]
        return first.rstrip(".") + "."
    return best.rstrip(".") + "."


if __name__ == "__main__":
    ctx = [
        "Set temperature to 0.0 for factual question answering.",
        "Chunk size determines retrieval precision.",
    ]
    print(generate_rag_answer("what temperature for factual QA", ctx,
                              GenerationConfig()))
