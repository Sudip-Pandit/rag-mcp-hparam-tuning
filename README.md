# RAG + MCP Hyperparameter Tuning

Runnable companion code for the article **[I Ran 50 Hyperparameter Experiments on a Production RAG System — Two Parameters Did 80% of the Work](https://levelup.gitconnected.com/i-ran-50-hyperparameter-experiments-on-a-production-rag-system-two-parameters-did-80-of-the-work-01c612422135)**.

Every hyperparameter discussed in the article is a real, tunable knob here. The whole repo **runs offline with zero API keys** (local hashing embeddings, an in-memory vector store, a deterministic planner). Drop in `OPENAI_API_KEY`, `PINECONE_API_KEY`, `COHERE_API_KEY`, or `ANTHROPIC_API_KEY` to switch any stage to live mode.

## Quick start

```bash
git clone https://github.com/Sudip-Pandit/rag-mcp-hyperparameter-tuning.git
cd rag-mcp-hparam-tuning

# Offline mode needs nothing installed. To run the whole tour:
python run_demo.py
```

That runs: synthetic data generation → indexing → a 20-config hyperparameter sweep → a ranked report → both case studies (HR Policy Bot and Support Ticket Agent).

For live API mode:

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
```

## The two parameters that matter most

Per the article, `similarity_threshold` and `chunk_size` carry most of the quality gain. Both are first-class fields in `rag_agent/config.py:RAGConfig` and are swept in `scripts/run_sweep.py`.

## Repository layout

```
rag-mcp-hparam-tuning/
├── rag_agent/         # Part 1: RAG tuning
│   ├── config.py        master RAGConfig (every tunable parameter)
│   ├── chunker.py       1.1 token-aware chunking + cost estimator
│   ├── embedder.py      1.2 embedding model/dimension (OpenAI or local)
│   ├── retriever.py     1.3 top_k, metric, similarity_threshold + vector store
│   ├── reranker.py      1.4 cross-encoder / Cohere two-stage rerank
│   ├── query_rewriter.py 1.5 NONE/HyDE/multi-query/step-back/fusion
│   ├── generator.py     1.6 temperature/top_p/max_tokens/penalties
│   └── pipeline.py      end-to-end RAG with cost + latency tracking
│
├── mcp_agent/         # Part 2: MCP tuning
│   ├── config.py        tool gating + agent budgets + guardrails
│   ├── tools.py         example ticket tools w/ risk levels
│   ├── router.py        2.1 confidence-gated tool selection
│   ├── guardrails.py    2.2/2.5 retry strategy + destructive-action gate
│   ├── memory.py        2.3 sliding memory window + summarization
│   └── agent.py         bounded agent loop (max_steps, token budget)
│
├── evaluation/        # Part 1.7 / Part 3
│   ├── ragas_eval.py    hit_rate, MRR, faithfulness triad
│   ├── latency_eval.py  P50/P95/P99
│   ├── cost_eval.py     per-query + projected monthly cost
│   └── report_generator.py  ranked sweep report
│
├── shared/
│   ├── cost_tracker.py     Part 3.1 cost model
│   ├── latency_tracker.py  Part 3.3 stage timing
│   └── logging_config.py   PII-redacting logger
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── index_documents.py
│   ├── run_sweep.py        Part 4.3 sweep (default 50 experiments)
│   └── benchmark.py        Part 5 case studies, before/after
│
├── experiments/
│   ├── sweep_config.yaml   canonical search space + A/B canary
│   └── results/            sweep_results.jsonl lands here
│
├── data/synthetic/         generated QA pairs + ground truth
├── run_demo.py             one-command tour
├── requirements.txt
└── .env.example
```

## Running pieces individually

```bash
# Each module has a __main__ smoke test:
python rag_agent/chunker.py
python -m rag_agent.pipeline
python -m mcp_agent.agent

# A bigger sweep (writes experiments/results/sweep_results.jsonl):
python scripts/run_sweep.py 50
python -m evaluation.report_generator
```

## Production checklist (from the article)

RAG: set `chunk_size` by document type, keep `chunk_overlap` at 15–20%, always set `similarity_threshold >= 0.6`, add a reranker after validating baseline, `temperature=0.0` for factual Q&A.

MCP: `temperature=0.0` for tool calls, define `stop_sequences`, set `max_steps` and `execution_budget_tokens`, gate destructive tools behind human confirmation, log every tool call.

## Live mode notes

- **OpenAI**: set `OPENAI_API_KEY`; embeddings and generation switch automatically.
- **Pinecone**: pass a real index object into `RAGPipeline(index=...)`; the `retrieve_chunks` contract is identical.
- **Cohere reranker**: `RerankerConfig(use_cohere=True)` with `COHERE_API_KEY`.
- **Anthropic MCP**: set `ANTHROPIC_API_KEY` to drive the agent with a live model (the offline planner stays available for deterministic tests).

## Author

Written by Sudip P. Read the full walkthrough in the [companion article](https://levelup.gitconnected.com/i-ran-50-hyperparameter-experiments-on-a-production-rag-system-two-parameters-did-80-of-the-work-01c612422135), and follow along on [Medium](https://medium.com/@banisusan045) for more on production RAG, MCP, and AI engineering.

## License

MIT. Use freely with attribution to the article.
