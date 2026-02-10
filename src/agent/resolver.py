from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .prompts import RESOLUTION_USER, SYSTEM_JSON_ONLY
from .schemas import Claim, Edge, EvidenceCard, Resolution
from .utils import get_llm


class ResolutionOutput(BaseModel):
    summary: str
    conditions: str | None = None
    leaning_claim_id: str | None = None


def _edge_id(edge: Edge) -> str:
    return f"{edge.src_claim_id}->{edge.dst_claim_id}"


def _component_edges(edges: List[Edge]) -> Dict[str, List[Edge]]:
    graph: Dict[str, List[Edge]] = defaultdict(list)
    for edge in edges:
        if edge.relation != "contradicts":
            continue
        graph[edge.src_claim_id].append(edge)
        graph[edge.dst_claim_id].append(edge)
    return graph


def _find_components(claim_ids: List[str], edges: List[Edge]) -> List[List[str]]:
    graph = _component_edges(edges)
    visited = set()
    components: List[List[str]] = []

    for cid in claim_ids:
        if cid in visited or cid not in graph:
            continue
        stack = [cid]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for edge in graph.get(cur, []):
                other = edge.dst_claim_id if edge.src_claim_id == cur else edge.src_claim_id
                if other not in visited:
                    stack.append(other)
        if comp:
            components.append(comp)
    return components


def resolve_contradictions(
    claims: List[Claim],
    edges: List[Edge],
    evidence_by_id: Dict[str, EvidenceCard],
    llm: Optional[object] = "auto",
) -> Tuple[List[Resolution], List[Edge]]:
    if llm == "auto":
        try:
            llm = get_llm()
        except Exception:
            llm = None

    claim_map = {c.id: c for c in claims}
    contradiction_edges = [e for e in edges if e.relation == "contradicts"]
    components = _find_components(list(claim_map.keys()), contradiction_edges)

    resolutions: List[Resolution] = []
    for idx, comp in enumerate(components, start=1):
        comp_edges = [
            e for e in contradiction_edges if e.src_claim_id in comp and e.dst_claim_id in comp
        ]
        weight_by_claim: Dict[str, float] = {}
        for cid in comp:
            claim = claim_map[cid]
            total = 0.0
            for eid in claim.supported_by:
                ev = evidence_by_id.get(eid)
                if ev:
                    total += ev.evidence_weight
            weight_by_claim[cid] = total

        claims_block = "\n".join([f"- {cid}: {claim_map[cid].statement}" for cid in comp])
        weights_block = "\n".join([f"- {cid}: {weight_by_claim[cid]:.2f}" for cid in comp])

        summary = "Contradictory claims detected."
        conditions = None
        leaning = None
        if llm is not None:
            prompt_tmpl = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_JSON_ONLY),
                    ("user", RESOLUTION_USER),
                ]
            )
            messages = prompt_tmpl.format_messages(
                claims_block=claims_block,
                weights_block=weights_block,
            )
            structured = llm.with_structured_output(ResolutionOutput)
            output = structured.invoke(messages)
            summary = output.summary
            conditions = output.conditions
            leaning = output.leaning_claim_id

        resolution_id = f"R{idx}"
        for edge in comp_edges:
            edge.resolution_id = resolution_id

        resolutions.append(
            Resolution(
                id=resolution_id,
                claim_ids=comp,
                edge_ids=[_edge_id(e) for e in comp_edges],
                summary=summary,
                conditions=conditions,
                leaning_claim_id=leaning,
                weight_by_claim=weight_by_claim,
            )
        )

    return resolutions, edges
