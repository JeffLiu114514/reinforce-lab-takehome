# PLAN.md - Claim Graph + Contradiction Finder (LangChain, Python)

This plan is written as **instructions for a coding agent** to implement a minimal-but-solid AI web research agent that:
- Accepts a research prompt
- Plans investigation angles
- Searches the web
- Retrieves + extracts evidence
- Builds a **Claim Graph** with **supports/contradicts** edges
- Produces a **Markdown research report** that **separates claims from evidence** and surfaces contradictions

The system should be implementable in ~1 day and runnable end-to-end via a single CLI command.

---

## 0) Scope: what to build (MVP)

### Must-have behaviors (per assignment)
1. Identify multiple research angles
2. Gather sources across academia + industry
3. Extract evidence for each claim type
4. Separate claims from evidence

### MVP definition
Given a prompt like:
> "What are the real-world risks and benefits of using synthetic data to train or fine-tune large language models? Focus on data quality, bias, and evaluation?"
The agent should output:
- `artifacts/ledger.json` (structured: sources, evidence cards, claims, graph edges)
- `artifacts/report.md` (human-readable research report with citations + a contradiction section)

---

## 1) Tech stack (pin these choices)

### Language & runtime
- Python 3.11+
- `uv` or `pip` + `venv` (choose one; prefer `uv` for speed)

### Environment management (required)
- **Conda is installed locally**. You MUST manage Python dependencies using **a new dedicated conda environment** for this repo.
- Create and activate the env (example):
  ```bash
  conda create -n research-agent-graph python=3.11 -y
  conda activate research-agent-graph
  ```
- Install dependencies via one of:
  - `pip install -e .` (preferred, uses `pyproject.toml`), or
  - `pip install -r requirements.txt` if you choose to generate one.
- Document these steps in `README.md` and keep the environment name consistent across docs and scripts.

### LLM + orchestration
- **LangChain** for LLM calls + tool wrappers
- Recommended: `langchain`, `langchain-community`, `langchain-openai` (or `langchain-anthropic` if preferred)

### Web search tool (no key required option)
- Default: `DuckDuckGoSearchRun` from `langchain_community.tools`
- Optional alternative (key-based): Tavily / SerpAPI / Bing. Provide an interface so it's swappable.

### HTML fetching + extraction
- `httpx` for HTTP
- `trafilatura` (best-effort content extraction) OR `readability-lxml` + `beautifulsoup4`
- Fallback: simple BeautifulSoup extraction if the above fails

### Data modeling
- `pydantic` for strict schemas

### Storage + caching
- Disk cache: `diskcache` (or a simple JSON+hash cache if minimizing deps)

### Testing
- `pytest` (+ `pytest-sugar` optional)

---

## 2) Repo layout (create exactly this structure)

```
research_agent/
  pyproject.toml
  README.md
  PLAN.md
  src/
    agent/
      __init__.py
      orchestrator.py
      planner.py
      search_tool.py
      fetcher.py
      extractor.py
      claim_builder.py
      contradiction.py
      report.py
      schemas.py
      prompts.py
      utils.py
    main.py
  tests/
    test_schemas.py
    test_contradiction.py
    test_smoke_end_to_end.py
  artifacts/
    .gitkeep
```

---

## 3) Core data structures (Pydantic schemas)

### 3.1 Source, Evidence, Claim, Graph Edge

Implement these in `src/agent/schemas.py`.

**Key design principle**: the report must be fully derivable from the structured ledger, and every claim must reference evidence IDs.

```python
from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal, Dict

SourceType = Literal["paper", "preprint", "blog", "report", "documentation", "news", "other"]
ClaimType  = Literal["data_quality", "bias", "evaluation", "privacy_security", "ops_risk"]

class Source(BaseModel):
    id: str
    url: HttpUrl
    title: str
    author: Optional[str] = None
    date: Optional[str] = None  # ISO if possible
    source_type: SourceType = "other"
    publisher: Optional[str] = None

class EvidenceCard(BaseModel):
    id: str
    source_id: str
    claim_types: List[ClaimType]
    snippet: str = Field(..., description="Short quote/passage extracted from the source")
    context: Optional[str] = Field(None, description="Optional extra context around snippet")
    reliability: int = Field(3, ge=1, le=5)
    notes: Optional[str] = None

class Claim(BaseModel):
    id: str
    claim_type: ClaimType
    statement: str
    polarity: Literal["pro", "con", "mixed", "neutral"] = "neutral"
    supported_by: List[str] = Field(default_factory=list, description="EvidenceCard ids")
    confidence: int = Field(3, ge=1, le=5)

class Edge(BaseModel):
    src_claim_id: str
    dst_claim_id: str
    relation: Literal["supports", "contradicts", "refines", "unrelated"]
    rationale: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)

class ClaimGraph(BaseModel):
    claims: List[Claim]
    edges: List[Edge]

class Ledger(BaseModel):
    prompt: str
    plan: Dict
    sources: List[Source]
    evidence: List[EvidenceCard]
    graph: ClaimGraph
    created_at: str
    version: str = "0.1"
```

### 3.2 Identifiers
- Use stable, human-readable IDs:
  - Sources: `S1`, `S2`, ...
  - Evidence: `E1`, `E2`, ...
  - Claims: `C1`, `C2`, ...
- Store in-memory maps for quick lookups:
  - `source_by_id`, `evidence_by_id`, `claim_by_id`

---

## 4) Pipeline (end-to-end steps)

Implement orchestration in `src/agent/orchestrator.py`. Keep each step pure and testable.

### Step 1 - Plan: angles + queries
**Input**: prompt  
**Output**: `plan` dict containing:
- `angles`: list[str]
- `queries`: list[str] (each query tied to an angle + claim types)
- `constraints`: include "data quality, bias, evaluation" explicitly for the test prompt

Implementation:
- In `planner.py`, call the LLM with a structured output schema (Pydantic / JSON mode).
- Force 4-6 angles and 2 queries per angle.

Planner output example:
```json
{
  "angles": [
    "Synthetic data effects on data quality and generalization",
    "Bias amplification vs bias mitigation using synthetic augmentation",
    "Evaluation pitfalls: synthetic wins but real-world fails",
    "Privacy/security tradeoffs: reduced PII vs new leakage modes"
  ],
  "queries": [
    {"q": "synthetic data training LLM evaluation real world performance", "claim_types": ["evaluation"]},
    {"q": "synthetic data bias mitigation LLM fine-tuning", "claim_types": ["bias"]},
    ...
  ]
}
```

### Step 2 - Search: gather candidate URLs
**Input**: queries  
**Output**: list of `SearchResult` objects: `{url, title, snippet}`

Implementation:
- `search_tool.py` exposes `search(query: str, k: int) -> list[dict]`.
- Default tool: DuckDuckGo via LangChain community tool.
- Add dedupe by canonical URL (strip tracking params).

Targets:
- `k=5` per query is enough for MVP.
- Total URLs cap ~25-35.

### Step 3 - Fetch + extract readable text
**Input**: URLs  
**Output**: `Source` objects + extracted `text` content

Implementation:
- `fetcher.py`:
  - Use `httpx` with timeouts and user-agent header.
  - Use `trafilatura.extract()` (or readability-lxml) to get main content.
  - Store raw HTML optionally (for debugging).
  - Cache by URL hash to avoid refetching.

Heuristics:
- If extracted text < 1,000 chars, mark source low quality; still keep but downweight reliability.

### Step 4 - Evidence extraction (EvidenceCards)
**Input**: extracted text + claim_types context  
**Output**: EvidenceCards with snippet + minimal metadata

Implementation in `extractor.py`:
- For each source, ask LLM to extract 1-2 evidence snippets relevant to the prompt, tagged with claim types.
- Use JSON-only output.
- Include a reliability estimate; for MVP this can be LLM-judged using a rubric:
  - 5: peer-reviewed paper / strong benchmark report
  - 4: reputable org report / high-quality technical blog with data
  - 3: informed blog post / secondary discussion
  - 2: marketing / opinion with weak support
  - 1: low credibility

Important constraint:
- Store **short snippets** (<= 400 chars) to keep report concise and reduce hallucination surface.

### Step 5 - Build claims from evidence
**Input**: EvidenceCards  
**Output**: Claims list (C1..Cn)

Implementation in `claim_builder.py`:
- Group EvidenceCards by `claim_type`.
- For each group, prompt the LLM:
  - Generate 3-5 claims total across all types (MVP)
  - Each claim must cite 1-3 evidence IDs.
  - Each claim has `polarity` (pro/con/mixed).
  - Provide `confidence` 1-5, based on strength + consistency.

Hard invariant:
- Reject any claim with `supported_by=[]`.

### Step 6 - Contradiction detection: edges in claim graph
**Input**: claims + evidence  
**Output**: edges `supports/contradicts/refines/unrelated`

Implementation in `contradiction.py`:
- Pairwise comparison can be O(n^2). For MVP, keep n small (<= 12 claims).
- Approach:
  1. Create embeddings for claim statements (optional) and only compare pairs above similarity threshold, OR
  2. For MVP simpler: compare claims within the same `claim_type` only (reduces pairs).
- For each pair, prompt LLM to label relation and provide a short rationale + evidence IDs if applicable.

Rule:
- `contradicts` if statements cannot both be true under the same conditions.
- `refines` if one adds conditions/nuance.
- `supports` if one implies/bolsters the other.
- `unrelated` otherwise.

### Step 7 - Generate Markdown report
**Input**: ledger  
**Output**: `artifacts/report.md`

Implementation in `report.py`:
Include sections:
1. Problem framing & assumptions
2. Research angles (from planner)
3. Findings by claim type:
   - For each claim: statement, confidence, evidence bullets (snippets + source)
4. Contradictions & tensions:
   - List contradicting claim pairs with rationale
5. Evidence appendix:
   - Table of sources + reliability
6. "Claims needing stronger evidence" (optional but recommended)

**Critical**: "Separate claims from evidence" is explicit:
- Use formatting like:

```md
### Claim C3 (bias) - Confidence: 3/5
**Claim:** Synthetic augmentation can reduce bias in low-resource subpopulations when targeted to coverage gaps.

**Evidence:**
- (E7, S4) "..snippet.."
- (E9, S6) "..snippet.."
```

---

## 5) Prompts (put in `src/agent/prompts.py`)

Create prompt templates with:
- System: "You are a careful research assistant. Do not invent sources. Output valid JSON only." - User: include the prompt, claim types, and small text chunks (truncate to safe max tokens).

### Truncation strategy
- For each source text: keep first 6k-10k characters, plus a few passages around keyword hits if you implement keyword indexing.
- If time: implement a `chunk_text(text, chunk_size=1500, overlap=200)` and run extraction per chunk, then merge/dedupe snippets.

---

## 6) Orchestrator wiring (LangChain)

### LLM configuration
In `src/agent/utils.py`:
- Support environment variables:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (default: `gpt-4.1-mini` or another)
- Create a single `get_llm()` that returns `ChatOpenAI(temperature=0)`.

### Execution entrypoint
In `src/main.py`:
- Parse CLI args:
  - `--prompt`
  - `--k_per_query` (default 5)
  - `--max_urls` (default 30)
  - `--out_dir` (default `artifacts/`)
- Call `run(prompt)` in orchestrator.

Expected command:
```bash
python -m src.main --prompt "What are the real-world risks and benefits of using synthetic data to train or fine-tune large language models? Focus on data quality, bias, and evaluation."
```

---

## 7) Minimal testing plan (must implement)

### `tests/test_schemas.py`
- Construct Ledger with 1 Source, 1 Evidence, 1 Claim, 0 edges.
- Validate Pydantic models.

### `tests/test_contradiction.py`
- Provide 3 synthetic claims:
  - C1: "Synthetic data improves generalization"
  - C2: "Synthetic data harms generalization in real deployment"
  - C3: "Synthetic data helps when distribution matches target"
- Assert the contradiction function returns at least one `contradicts` or `refines`.

### `tests/test_smoke_end_to_end.py`
- Use a short prompt (or the provided test prompt).
- Run orchestrator with `max_urls=5` and assert:
  - `report.md` exists
  - `ledger.json` exists
  - At least 2 claim types present
  - Every claim has >=1 evidence id

If external web is flaky, allow a `--offline` mode later; for MVP, just keep the smoke test optional or marked `@pytest.mark.integration`.

---

## 8) Quality & robustness guardrails (add these)

1. **Deduplication**
   - Dedupe sources by normalized URL.
   - Dedupe evidence snippets by hash of snippet text.

2. **Hallucination prevention**
   - Never allow LLM to "invent" citations: claims must cite extracted EvidenceCards only.
   - Evidence snippets must be verbatim or near-verbatim from the extracted text.

3. **Timeouts & failures**
   - If fetch fails, skip URL and continue.
   - If extraction fails for a source, keep Source but no EvidenceCards.

4. **Logging**
   - Print progress with counts: urls fetched, evidence extracted, claims created, contradictions found.

---

## 9) Implementation checklist (coding agent steps)

Follow this order; do not skip.

### Phase A - Scaffolding (30-45 min)
- [ ] Create repo layout
- [ ] Add `pyproject.toml` with pinned deps
- [ ] Add CLI entrypoint in `src/main.py`

### Phase B - Data layer (30 min)
- [ ] Implement `schemas.py`
- [ ] Add schema tests

### Phase C - Tools + retrieval (60-90 min)
- [ ] Implement `search_tool.py` (DuckDuckGo default)
- [ ] Implement `fetcher.py` with caching + extraction

### Phase D - LLM steps (90-120 min)
- [ ] Implement `planner.py` (structured plan output)
- [ ] Implement `extractor.py` (EvidenceCards)
- [ ] Implement `claim_builder.py` (Claims from EvidenceCards)

### Phase E - Graph + contradiction (60-90 min)
- [ ] Implement `contradiction.py` (edge labeling within claim_type groups)

### Phase F - Reporting (45-60 min)
- [ ] Implement `report.py` to produce Markdown with strict claims/evidence separation

### Phase G - Integration + smoke run (30-60 min)
- [ ] Implement orchestrator wiring
- [ ] Run the provided test prompt
- [ ] Inspect outputs, fix obvious issues

---

## 10) Acceptance criteria (what "done" means)

The system is acceptable if:
- Running one CLI command produces both `ledger.json` and `report.md`.
- Report contains:
  - angles
  - findings grouped by claim type
  - explicit claim blocks with evidence bullets
  - a contradictions section
- Ledger satisfies invariants:
  - every Claim has >=1 EvidenceCard id
  - every EvidenceCard references a valid Source id

---

## 11) README notes (include in repo)

In `README.md` include:
- Setup instructions
- Env vars
- Example command
- Where outputs go
- Limitations (e.g., web noise, extraction quality)
- How to extend search providers

---

## 12) Optional, low-cost "wow" additions (still within 1 day)
Pick at most one:
- Add a simple Mermaid diagram of the pipeline into `report.md`.
- Add a "Contradiction clusters" view (group contradictory claims by topic).
- Add a "Coverage score" (counts of claims per type: quality/bias/evaluation).

(Do not expand beyond this in the MVP build.)