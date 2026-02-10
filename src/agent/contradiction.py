from __future__ import annotations

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .prompts import RELATION_USER, SYSTEM_JSON_ONLY
from .schemas import Claim, Edge
from .utils import get_llm


class RelationOutput(BaseModel):
    relation: str
    rationale: str | None = None
    evidence_ids: List[str] = []


def _heuristic_relation(claim_a: Claim, claim_b: Claim) -> str:
    if claim_a.polarity in {"pro", "con"} and claim_b.polarity in {"pro", "con"}:
        if claim_a.polarity != claim_b.polarity:
            return "contradicts"
    a = claim_a.statement.lower()
    b = claim_b.statement.lower()
    if "when" in a or "if" in a or "when" in b or "if" in b:
        return "refines"
    return "unrelated"


def build_edges(claims: List[Claim], llm: Optional[object] = "auto") -> List[Edge]:
    if llm == "auto":
        try:
            llm = get_llm()
        except Exception:
            llm = None

    edges: List[Edge] = []
    for i, claim_a in enumerate(claims):
        for claim_b in claims[i + 1 :]:
            if claim_a.claim_type != claim_b.claim_type:
                continue
            if llm is None:
                relation = _heuristic_relation(claim_a, claim_b)
                edges.append(
                    Edge(
                        src_claim_id=claim_a.id,
                        dst_claim_id=claim_b.id,
                        relation=relation,
                        rationale=None,
                        evidence_ids=[],
                    )
                )
                continue

            prompt_tmpl = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_JSON_ONLY),
                    ("user", RELATION_USER),
                ]
            )
            messages = prompt_tmpl.format_messages(
                claim_a=claim_a.statement,
                claim_b=claim_b.statement,
            )
            structured = llm.with_structured_output(RelationOutput)
            output = structured.invoke(messages)
            edges.append(
                Edge(
                    src_claim_id=claim_a.id,
                    dst_claim_id=claim_b.id,
                    relation=output.relation,
                    rationale=output.rationale,
                    evidence_ids=output.evidence_ids or [],
                )
            )
    return edges

