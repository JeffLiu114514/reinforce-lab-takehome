from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .schemas import Claim, Edge, EvidenceCard, Source


def _proxy_labels(claims: List[Claim], edges: List[Edge]) -> List[int]:
    contradicted = set()
    for edge in edges:
        if edge.relation != "contradicts":
            continue
        contradicted.add(edge.src_claim_id)
        contradicted.add(edge.dst_claim_id)

    labels = []
    for claim in claims:
        labels.append(1 if (claim.id not in contradicted and len(claim.supported_by) >= 2) else 0)
    return labels


def _brier_score(probs: List[float], labels: List[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def _ece(probs: List[float], labels: List[int], bins: int = 10) -> float:
    if not probs:
        return 0.0
    n = len(probs)
    ece = 0.0
    for b in range(bins):
        low = b / bins
        high = (b + 1) / bins
        idxs = [i for i, p in enumerate(probs) if (low <= p < high) or (b == bins - 1 and p == 1.0)]
        if not idxs:
            continue
        acc = sum(labels[i] for i in idxs) / len(idxs)
        conf = sum(probs[i] for i in idxs) / len(idxs)
        ece += (len(idxs) / n) * abs(acc - conf)
    return ece


def compute_metrics(
    claims: List[Claim],
    edges: List[Edge],
    evidence: List[EvidenceCard],
    sources: List[Source],
    provider_stats: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}

    total_claims = len(claims)
    supported = sum(1 for c in claims if c.supported_by)
    metrics["supported_claim_rate"] = supported / total_claims if total_claims else 0.0

    providers = {s.provider for s in sources if s.provider}
    publishers = {s.publisher for s in sources if s.publisher}
    metrics["evidence_diversity"] = {
        "providers": len(providers),
        "publishers": len(publishers),
    }

    claim_type_counts = Counter([c.claim_type for c in claims])
    metrics["claim_type_coverage"] = dict(claim_type_counts)

    evaluated_pairs = len(edges)
    contradictions = sum(1 for e in edges if e.relation == "contradicts")
    metrics["contradiction_density"] = contradictions / evaluated_pairs if evaluated_pairs else 0.0

    weak = sum(1 for c in claims if c.confidence_score < 0.4)
    metrics["weak_evidence_rate"] = weak / total_claims if total_claims else 0.0
    metrics["confidence_distribution"] = dict(Counter([c.confidence for c in claims]))
    metrics["avg_confidence_score"] = (
        sum(c.confidence_score for c in claims) / total_claims if total_claims else 0.0
    )

    probs = [max(0.0, min(1.0, c.confidence_score)) for c in claims]
    labels = _proxy_labels(claims, edges)
    metrics["ece"] = _ece(probs, labels, bins=10)
    metrics["brier"] = _brier_score(probs, labels)
    metrics["calibration_label"] = "proxy_self_consistency"
    metrics["provider_health"] = provider_stats or {}

    return metrics
