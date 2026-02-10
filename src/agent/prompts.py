SYSTEM_JSON_ONLY = "You are a careful research assistant. Do not invent sources. Output valid JSON only."

PLANNER_USER = """
You are planning research angles and search queries.
Prompt: {prompt}

Requirements:
- Provide 4 to 6 distinct angles.
- Provide 2 queries per angle.
- Ensure queries explicitly mention data quality, bias, and evaluation when relevant.
- For each query, include provider_hint: one of academic, industry, general.
- Output JSON with keys: angles (list of strings), queries (list of objects with q, claim_types, provider_hint), constraints (list).
"""

EVIDENCE_USER = """
Extract 1 to 2 short evidence snippets relevant to the research prompt and claim types.
Prompt: {prompt}
Claim types of interest: {claim_types}
Source text (truncated):
"""

CLAIM_USER = """
Based on the evidence cards, generate 3 to 6 concise claims total across all types.
Each claim must cite 1 to 3 evidence ids.
Evidence cards:
{evidence_json}

Output JSON: {{"claims": [{{"claim_type": ..., "statement": ..., "polarity": ..., "supported_by": [...], "confidence": ...}}]}}
"""

RELATION_USER = """
Determine the relationship between two claims.
Claim A: {claim_a}
Claim B: {claim_b}

Choose one relation: supports, contradicts, refines, unrelated.
Provide a short rationale and relevant evidence ids if applicable.
Output JSON: {{"relation": "...", "rationale": "...", "evidence_ids": [...]}}
"""

RESOLUTION_USER = """
You are resolving a contradiction cluster.
Conflicting claims:
{claims_block}

Evidence weights by claim:
{weights_block}

Tasks:
- Summarize the conflict.
- State conditions under which both could be true, if any.
- If incompatible, choose the better-supported claim id; otherwise return null.
Output JSON: {{"summary": "...", "conditions": "...", "leaning_claim_id": null or "C#"}}
"""

