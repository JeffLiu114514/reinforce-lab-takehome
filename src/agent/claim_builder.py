from __future__ import annotations

import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .prompts import CLAIM_USER, SYSTEM_JSON_ONLY
from .schemas import Claim, EvidenceCard
from .utils import get_llm


class ClaimItem(BaseModel):
    claim_type: str
    statement: str
    polarity: str = "neutral"
    supported_by: List[str] = Field(default_factory=list)
    confidence: int = Field(3, ge=1, le=5)


class ClaimsOutput(BaseModel):
    claims: List[ClaimItem]


def _normalize_polarity(value: str) -> str:
    if not value:
        return "neutral"
    norm = value.strip().lower()
    mapping = {
        "positive": "pro",
        "pos": "pro",
        "pro": "pro",
        "negative": "con",
        "neg": "con",
        "con": "con",
        "mixed": "mixed",
        "both": "mixed",
        "neutral": "neutral",
        "none": "neutral",
    }
    return mapping.get(norm, "neutral")


def build_claims(evidence: List[EvidenceCard]) -> List[Claim]:
    llm = get_llm()

    evidence_json = json.dumps([e.model_dump() for e in evidence], indent=2)
    valid_ids = {e.id for e in evidence}
    prompt_tmpl = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_JSON_ONLY),
            ("user", CLAIM_USER),
        ]
    )
    messages = prompt_tmpl.format_messages(evidence_json=evidence_json)
    structured = llm.with_structured_output(ClaimsOutput)
    output = structured.invoke(messages)

    claims: List[Claim] = []
    for idx, item in enumerate(output.claims, start=1):
        supported = [eid for eid in item.supported_by if eid in valid_ids]
        if not supported:
            continue
        claims.append(
            Claim(
                id=f"C{idx}",
                claim_type=item.claim_type,  # validated by Claim model
                statement=item.statement,
                polarity=_normalize_polarity(item.polarity),
                supported_by=supported,
                confidence=item.confidence,
            )
        )
    return claims

