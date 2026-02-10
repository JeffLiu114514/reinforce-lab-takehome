# SDE Take-Home: AI Agent for Web Research

Build a minimal-but-solid research agent that plans, searches, extracts evidence, and produces a Markdown report. This repo includes a working CLI, a simple Web UI, and structured artifacts (`ledger.json`, `trace.json`, and a Mermaid claim graph).

## Take-Home Goals (Mapped to This Repo)

From the prompt in the take-home PDF:

- Accepts a research prompt: CLI `--prompt` and Web UI prompt box.
- Plans how to investigate: LLM planner produces research angles + query specs.
- Searches the web: provider router hits DuckDuckGo and (when available) academic providers.
- Retrieves and extracts evidence: fetch pages, extract text, pull verified snippets.
- Produces a Markdown report: `artifacts/report.md` plus structured ledger/trace.
- Beyond minimal: source/evidence weighting, snippet verification, claim clustering, contradiction graph + resolution objects, metrics, and Mermaid export.

## Problem Framing & Assumptions

The goal is to turn an open-ended research prompt into a set of evidence-backed claims, with enough structure to debug and extend the system.

Assumptions and non-goals:
- Assumes network access and valid `GOOGLE_API_KEY` for the LLM.
- Assumes some URLs are fetchable and contain extractable text; the fetcher uses best-effort extraction and caching.
- Does not guarantee correctness; confidence is a heuristic score intended for prioritization and review, not a calibrated truth probability.
- The system is evidence-grounded by construction: claims must reference evidence IDs extracted from retrieved text.

## Quick Start

### 1) Environment

This project targets Python 3.11+.

Conda (recommended):
```bash
conda create -n research-agent-graph python=3.11 -y
conda activate research-agent-graph
```

Install:
```bash
cd research_agent
pip install -e .
```

### 2) API Key

Set:
- `GOOGLE_API_KEY` (required)
- `GOOGLE_MODEL` (optional, default: `gemini-2.5-flash`)

Examples:

PowerShell:
```powershell
$env:GOOGLE_API_KEY="..."
$env:GOOGLE_MODEL="gemini-2.5-flash"
```

bash/zsh:
```bash
export GOOGLE_API_KEY="..."
export GOOGLE_MODEL="gemini-2.5-flash"
```

Optional (improves academic retrieval reliability):
- `SEMANTIC_SCHOLAR_API_KEY`

## Run (CLI)

From `research_agent/`:
```bash
python -m src.main --prompt "What are the real-world risks and benefits of using synthetic data to train or fine-tune large language models? Focus on data quality, bias, and evaluation." --config "config/source_weights.json"
```

Outputs go to `research_agent/artifacts/` by default:
- `research_agent/artifacts/report.md`
- `research_agent/artifacts/ledger.json`
- `research_agent/artifacts/trace.json`
- `research_agent/artifacts/graph.mmd`

Key knobs:
- `--k_per_query` (default 8)
- `--max_urls` (default 30)
- `--out_dir` (default `artifacts`)
- `--config` (default `config/source_weights.json`)

## Run (Web UI)

From `research_agent/`:
```bash
python -m src.web --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

UI features:
- live progress updates (same stage messages as CLI)
- conclusion summary (claims + confidence)
- rendered claim graph (Mermaid)
- artifact viewers for `report.md`, `graph.mmd`, `ledger.json`, `trace.json`

## Workflow (How the Agent Achieves the Goal)

At a high level, the system is a staged pipeline where every report claim is backed by evidence IDs (and all artifacts are derivable from `ledger.json`):

1. Plan: generate research angles and query specs (with provider hints).
2. Search: route each query to providers; dedupe URLs and preserve best metadata per URL.
3. Fetch: retrieve pages and extract main text (with caching).
4. Evidence: extract short snippets and verify they exist in the page text (exact or fuzzy).
5. Claims: synthesize claims from evidence cards; every claim must cite evidence IDs.
6. Canonicalize: cluster near-duplicate claims to reduce graph complexity.
7. Graph: infer claim-to-claim relations and find contradictions.
8. Resolve: group contradiction components and (optionally) summarize resolutions.
9. Score: compute interpretable confidence components and a calibrated 1-5 rating.
10. Write artifacts: `ledger.json`, `trace.json`, `report.md`, `graph.mmd`.

### Architecture Diagram

```mermaid
flowchart LR
  A[Prompt] --> B[Planner]
  B --> C[Query Specs + Provider Hints]
  C --> D[Search Router]
  D --> E[Search Providers]
  E --> F[URLs + Metadata]
  F --> G[Fetcher + Text Extraction]
  G --> H[Evidence Extractor + Verification]
  H --> I[Evidence Cards]
  I --> J[Claim Builder]
  J --> K[Claim Clustering]
  K --> L[Claim Graph (Edges)]
  L --> M[Contradiction Resolver]
  M --> N[Metrics + Confidence]
  N --> O[Ledger + Trace]
  O --> P[Report.md + Graph.mmd]
```

### Where to Look in Code

Everything lives under `research_agent/src/agent/`:
- Planning: `planner.py`
- Search + provider routing: `search_providers.py`
- Fetch + text extraction: `fetcher.py`
- Evidence extraction + verification: `extractor.py`
- Claim synthesis: `claim_builder.py`
- Claim clustering: `claim_cluster.py`
- Edge/contradiction detection: `contradiction.py`
- Contradiction resolution objects: `resolver.py`
- Confidence + weighting: `weights.py`
- Metrics: `metrics.py`
- Report rendering: `report.py`
- Mermaid export: `mermaid.py`
- Orchestration + artifact writing: `orchestrator.py`

## Tradeoffs (Quality vs Cost, Simplicity vs Scale)

- Evidence-grounded by construction: claims are generated only from extracted evidence IDs, and snippets are verified against page text.
- Provider reliability vs complexity: multiple providers improve diversity, but rate limits and availability vary (provider health is recorded in `trace.json` and metrics).
- Confidence is heuristic (not true correctness probability): it combines evidence strength, diversity, verification, and contradiction penalties, then uses a simple quantile calibration for 1-5 ratings.
- Scale is bounded intentionally: limits like `max_urls`, snippet caps, and clustering keep runtime and cost predictable without building a full distributed crawler.

## Presentation Test Prompt

The take-home test prompt is supported directly:
> What are the real-world risks and benefits of using synthetic data to train or fine-tune large language models? Focus on data quality, bias, and evaluation.

Expected behavior checklist:
- multiple angles + query specs
- sources across academia and industry (subject to provider rate limits)
- evidence extracted per claim type
- claims separated from evidence (claims reference evidence IDs)

## Testing

From `research_agent/`:
```bash
pytest -q
```

Integration smoke test (hits external web + LLM) requires `GOOGLE_API_KEY`:
```bash
pytest -q -m integration
```

## Future Extensions (If Given Unlimited Time)

- Retrieval quality: hybrid search (BM25 + embeddings), fusion (RRF), and reranking with a small cross-encoder.
- Verification: add a claim verifier (support/refute/insufficient-evidence) and propagate verification into confidence and contradiction resolution.
- Better calibration: collect a labeled evaluation set and apply temperature scaling or isotonic regression instead of quantile bins.
- Stronger source typing: classify source type (paper/blog/news/etc.) more reliably than URL heuristics.
- Offline corpora: ingest a provided corpus (PDFs/HTML dumps), add a local index, and run the pipeline without network access.
- UI and ops: websocket streaming logs, multi-run history, artifact browser, persistence in SQLite, and a queue/worker model for concurrent users.

## Notes / Limitations

- Web sources vary in quality; extraction can be noisy.
- Academic providers can rate-limit (especially without API keys); the system degrades gracefully and records provider health.
- Mermaid rendering in the Web UI uses a CDN; if offline, the UI falls back to showing the raw `graph.mmd` text.
