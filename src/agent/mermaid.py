from __future__ import annotations

import textwrap
from typing import List

from .schemas import Claim, Edge


NODE_CLASS_BY_CLAIM_TYPE = {
    "data_quality": "claim_data_quality",
    "bias": "claim_bias",
    "evaluation": "claim_evaluation",
    "privacy_security": "claim_privacy_security",
    "ops_risk": "claim_ops_risk",
}

EDGE_COLOR_BY_RELATION = {
    "supports": "#2e7d32",
    "contradicts": "#c62828",
    "refines": "#ef6c00",
    "unrelated": "#616161",
}


def _clean_label(text: str) -> str:
    cleaned = text.replace("\n", " ").replace('"', "'").replace("[", "(").replace("]", ")")
    return cleaned.strip()


def _wrap_label(text: str, width: int = 72) -> str:
    wrapped = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if not wrapped:
        return text
    return "<br/>".join(wrapped)


def _border_color(confidence: int) -> str:
    if confidence >= 4:
        return "#1b5e20"
    if confidence == 3:
        return "#ef6c00"
    return "#b71c1c"


def _border_width(confidence: int) -> int:
    if confidence >= 5:
        return 5
    if confidence == 4:
        return 4
    if confidence == 3:
        return 3
    return 2


def render_claim_graph(claims: List[Claim], edges: List[Edge]) -> str:
    lines = ["graph TD"]
    for claim in claims:
        raw_label = _clean_label(f"{claim.id}: {claim.statement}")
        label = _wrap_label(raw_label)
        node_class = NODE_CLASS_BY_CLAIM_TYPE.get(claim.claim_type, "claim_other")
        lines.append(f'{claim.id}["{label}"]:::{node_class}')
        stroke = _border_color(claim.confidence)
        width = _border_width(claim.confidence)
        lines.append(f"style {claim.id} stroke:{stroke},stroke-width:{width}px")
        # Works in Mermaid-enabled markdown renderers that support click directives.
        lines.append(f'click {claim.id} "#claim-{claim.id.lower()}" "Open claim details"')

    for idx, edge in enumerate(edges):
        label = edge.relation
        lines.append(f"{edge.src_claim_id} -->|{label}| {edge.dst_claim_id}")
        color = EDGE_COLOR_BY_RELATION.get(edge.relation, EDGE_COLOR_BY_RELATION["unrelated"])
        lines.append(f"linkStyle {idx} stroke:{color},stroke-width:2px")

    lines.append("")
    lines.append("classDef claim_data_quality fill:#e3f2fd,stroke:#1e88e5,color:#0d47a1")
    lines.append("classDef claim_bias fill:#fce4ec,stroke:#d81b60,color:#880e4f")
    lines.append("classDef claim_evaluation fill:#e8f5e9,stroke:#43a047,color:#1b5e20")
    lines.append("classDef claim_privacy_security fill:#fff3e0,stroke:#fb8c00,color:#e65100")
    lines.append("classDef claim_ops_risk fill:#f3e5f5,stroke:#8e24aa,color:#4a148c")
    lines.append("classDef claim_other fill:#eceff1,stroke:#607d8b,color:#263238")
    lines.append("")
    return "\n".join(lines)
