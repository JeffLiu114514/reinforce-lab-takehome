from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .schemas import Claim, EvidenceCard, Ledger, Source


def _source_lookup(sources: List[Source]) -> Dict[str, Source]:
    return {s.id: s for s in sources}


def _evidence_lookup(evidence: List[EvidenceCard]) -> Dict[str, EvidenceCard]:
    return {e.id: e for e in evidence}


def render_report(ledger: Ledger) -> str:
    sources = _source_lookup(ledger.sources)
    evidence_map = _evidence_lookup(ledger.evidence)

    lines: List[str] = []
    lines.append("# Research Report")
    lines.append("")
    lines.append("## Problem framing & assumptions")
    lines.append(f"Prompt: {ledger.prompt}")
    lines.append("")
    lines.append("## Research angles")
    for angle in ledger.plan.get("angles", []):
        lines.append(f"- {angle}")
    lines.append("")
    constraints = ledger.plan.get("constraints", [])
    if constraints:
        lines.append("## Constraints")
        for c in constraints:
            lines.append(f"- {c}")
        lines.append("")

    lines.append("## Findings by claim type")
    claims_by_type: Dict[str, List[Claim]] = defaultdict(list)
    for claim in ledger.graph.claims:
        claims_by_type[claim.claim_type].append(claim)

    for claim_type, claims in claims_by_type.items():
        lines.append("")
        lines.append(f"### {claim_type}")
        for claim in claims:
            lines.append("")
            lines.append(f"<a id=\"claim-{claim.id.lower()}\"></a>")
            lines.append(f"#### Claim {claim.id} ({claim.claim_type}) - Confidence: {claim.confidence}/5")
            lines.append(f"**Claim:** {claim.statement}")
            if claim.aliases:
                lines.append(f"**Aliases:** {' | '.join(claim.aliases)}")
            if claim.confidence_components:
                comp = claim.confidence_components
                lines.append(
                    "**Confidence components:** "
                    f"strength={comp.get('strength', 0.0):.2f}, "
                    f"diversity={comp.get('diversity', 0.0):.2f}, "
                    f"verification={comp.get('verification', 0.0):.2f}, "
                    f"conflict_penalty={comp.get('conflict_penalty', 1.0):.2f}"
                )
            if claim.needs_more_evidence:
                lines.append("**Status:** needs_more_evidence")
            lines.append("")
            lines.append("**Evidence:**")
            for ev_id in claim.supported_by:
                ev = evidence_map.get(ev_id)
                if not ev:
                    continue
                src = sources.get(ev.source_id)
                src_title = src.title if src else ev.source_id
                verify_flag = "verified" if ev.verified else "unverified"
                method = ev.verification_method or "none"
                lines.append(
                    f"- ({ev.id}, {ev.source_id}) {ev.snippet} - {src_title} "
                    f"[weight: {ev.evidence_weight:.2f}, {verify_flag}, method: {method}]"
                )

    lines.append("")
    lines.append("## Contradictions & tensions")
    contradiction_edges = [e for e in ledger.graph.edges if e.relation == "contradicts"]
    if contradiction_edges:
        for edge in contradiction_edges:
            lines.append(
                f"- {edge.src_claim_id} vs {edge.dst_claim_id}: {edge.rationale or 'Contradiction detected.'}"
                f"{' (resolution: ' + edge.resolution_id + ')' if edge.resolution_id else ''}"
            )
    else:
        lines.append("- None detected.")

    lines.append("")
    lines.append("## Contradiction resolutions")
    if ledger.resolutions:
        for res in ledger.resolutions:
            lines.append(f"- {res.id}: {res.summary}")
            if res.conditions:
                lines.append(f"Conditions ({res.id}): {res.conditions}")
            if res.leaning_claim_id:
                lines.append(f"Leaning ({res.id}): {res.leaning_claim_id}")
            if res.weight_by_claim:
                lines.append(f"Weights ({res.id}): {res.weight_by_claim}")
    else:
        lines.append("- None.")

    lines.append("")
    lines.append("## Evidence appendix")
    for source in ledger.sources:
        provider = source.provider or "unknown"
        stype = source.source_type
        lines.append(
            f"- {source.id}: {source.title} ({source.url}) "
            f"[provider: {provider}, type: {stype}, source_weight: {source.source_weight:.2f}]"
        )

    lines.append("")
    lines.append("## Claims needing stronger evidence")
    weak = [c for c in ledger.graph.claims if c.confidence <= 2 or c.needs_more_evidence]
    if not weak:
        lines.append("- None flagged.")
    else:
        for claim in weak:
            lines.append(f"- {claim.id}: {claim.statement}")

    lines.append("")
    lines.append("## Evaluation metrics")
    if ledger.metrics:
        for key, value in ledger.metrics.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- None.")

    lines.append("")
    return "\n".join(lines)
