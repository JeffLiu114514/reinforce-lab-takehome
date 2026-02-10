from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from .prompts import EVIDENCE_USER, SYSTEM_JSON_ONLY
from .schemas import EvidenceCard, ClaimType, Source
from .utils import chunk_text, get_llm, hash_text, sanitize_whitespace, truncate_text


class EvidenceItem(BaseModel):
    claim_types: List[ClaimType]
    snippet: str
    context: str | None = None
    reliability: int = Field(3, ge=1, le=5)
    notes: str | None = None


class EvidenceOutput(BaseModel):
    evidence: List[EvidenceItem]


def _verify_snippet(text: str, snippet: str, threshold: int) -> Tuple[bool, str | None, float | None]:
    if not text or not snippet:
        return False, "none", None
    norm_text = sanitize_whitespace(text).lower()
    norm_snip = sanitize_whitespace(snippet).lower()
    if norm_snip in norm_text:
        return True, "exact", 100.0
    score = float(fuzz.partial_ratio(norm_snip, norm_text))
    if score >= threshold:
        return True, "fuzzy", score
    return False, "none", score


def extract_evidence(
    prompt: str,
    source: Source,
    text: str,
    max_snippet_chars: int = 400,
    verification_cfg: Dict | None = None,
) -> List[EvidenceCard]:
    llm = get_llm()
    prompt_tmpl = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_JSON_ONLY),
            ("user", EVIDENCE_USER + "{text}"),
        ]
    )

    evidence_cards: List[EvidenceCard] = []
    seen_hashes = set()

    verification_cfg = verification_cfg or {}
    threshold = int(verification_cfg.get("fuzzy_threshold", 85))
    keep_unverified = bool(verification_cfg.get("keep_unverified", False))

    truncated = truncate_text(text, max_chars=8000)
    chunks = chunk_text(truncated, chunk_size=1500, overlap=200, max_chunks=1)

    for chunk in chunks[:4]:
        messages = prompt_tmpl.format_messages(
            prompt=prompt,
            claim_types=["data_quality", "bias", "evaluation", "privacy_security", "ops_risk"],
            text=chunk,
        )
        structured = llm.with_structured_output(EvidenceOutput)
        output = structured.invoke(messages)
        for item in output.evidence:
            snippet = sanitize_whitespace(item.snippet)[:max_snippet_chars]
            if not snippet:
                continue
            h = hash_text(snippet)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            verified, method, score = _verify_snippet(text, snippet, threshold)
            if not verified and not keep_unverified:
                continue
            reliability = item.reliability if verified else 1
            evidence_cards.append(
                EvidenceCard(
                    id="",
                    source_id=source.id,
                    claim_types=item.claim_types,
                    snippet=snippet,
                    context=sanitize_whitespace(item.context) if item.context else None,
                    reliability=reliability,
                    notes=item.notes,
                    verified=verified,
                    verification_method=method,
                    verification_score=score,
                )
            )
        if len(evidence_cards) >= 3:
            break

    return evidence_cards

