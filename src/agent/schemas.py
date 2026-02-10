from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl

SourceType = Literal[
    "paper",
    "preprint",
    "blog",
    "report",
    "documentation",
    "news",
    "other",
]

ClaimType = Literal[
    "data_quality",
    "bias",
    "evaluation",
    "privacy_security",
    "ops_risk",
]


class Source(BaseModel):
    id: str
    url: HttpUrl
    title: str
    author: Optional[str] = None
    date: Optional[str] = None  # ISO if possible
    source_type: SourceType = "other"
    publisher: Optional[str] = None
    provider: Optional[str] = None
    provider_status: Optional[str] = None
    provider_error_code: Optional[str] = None
    domain: Optional[str] = None
    source_weight: float = 1.0


class EvidenceCard(BaseModel):
    id: str
    source_id: str
    claim_types: List[ClaimType]
    snippet: str = Field(..., description="Short quote/passage extracted from the source")
    context: Optional[str] = Field(None, description="Optional extra context around snippet")
    reliability: int = Field(3, ge=1, le=5)
    notes: Optional[str] = None
    evidence_weight: float = 0.0
    verified: bool = False
    verification_method: Optional[str] = None
    verification_score: Optional[float] = None


class Claim(BaseModel):
    id: str
    claim_type: ClaimType
    statement: str
    polarity: Literal["pro", "con", "mixed", "neutral"] = "neutral"
    supported_by: List[str] = Field(
        default_factory=list, description="EvidenceCard ids"
    )
    confidence: int = Field(3, ge=1, le=5)
    confidence_score: float = 0.0
    confidence_components: Dict[str, float] = Field(default_factory=dict)
    needs_more_evidence: bool = False
    aliases: List[str] = Field(default_factory=list)


class Edge(BaseModel):
    src_claim_id: str
    dst_claim_id: str
    relation: Literal["supports", "contradicts", "refines", "unrelated"]
    rationale: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    resolution_id: Optional[str] = None


class ClaimGraph(BaseModel):
    claims: List[Claim]
    edges: List[Edge]


class Resolution(BaseModel):
    id: str
    claim_ids: List[str]
    edge_ids: List[str]
    summary: str
    conditions: Optional[str] = None
    leaning_claim_id: Optional[str] = None
    weight_by_claim: Dict[str, float] = Field(default_factory=dict)


class Ledger(BaseModel):
    prompt: str
    plan: Dict
    sources: List[Source]
    evidence: List[EvidenceCard]
    graph: ClaimGraph
    resolutions: List[Resolution] = Field(default_factory=list)
    metrics: Dict[str, object] = Field(default_factory=dict)
    created_at: str
    version: str = "0.1"

