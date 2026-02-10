from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def compute_source_weight(
    provider: str | None,
    source_type: str | None,
    domain: str | None,
    config: Dict,
) -> float:
    provider_weights = config.get("provider_weights", {})
    source_type_weights = config.get("source_type_weights", {})
    domain_overrides = config.get("domain_overrides", {})

    provider_weight = provider_weights.get(provider or "", provider_weights.get("duckduckgo", 0.7))
    type_weight = source_type_weights.get(source_type or "other", source_type_weights.get("other", 0.5))
    domain_weight = domain_overrides.get(domain or "", 1.0)

    weight = provider_weight * type_weight * domain_weight
    return clamp(weight, 0.1, 1.0)


def compute_evidence_weight(source_weight: float, reliability: int) -> float:
    return clamp(source_weight * (float(reliability) / 5.0), 0.0, 1.0)


def compute_evidence_strength(evidence_weights: Iterable[float], redundancy_decay: float = 0.85) -> float:
    """Saturating accumulation with diminishing contribution from redundant evidence."""
    sorted_weights = sorted((clamp(float(w), 0.0, 1.0) for w in evidence_weights), reverse=True)
    if not sorted_weights:
        return 0.0
    remaining = 1.0
    for idx, weight in enumerate(sorted_weights):
        adjusted = clamp(weight * (redundancy_decay ** idx), 0.0, 1.0)
        remaining *= (1.0 - adjusted)
    return clamp(1.0 - remaining, 0.0, 1.0)


def compute_claim_confidence_components(
    evidence_weights: Sequence[float],
    evidence_source_ids: Sequence[str],
    source_provider_by_id: Dict[str, str],
    source_publisher_by_id: Dict[str, str],
    verification_scores: Sequence[float],
    conflict_penalty: float = 1.0,
) -> Dict[str, float]:
    """Return a normalized, interpretable confidence component breakdown."""
    strength = compute_evidence_strength(evidence_weights)

    unique_sources = len({sid for sid in evidence_source_ids if sid})
    provider_values = {
        source_provider_by_id.get(sid, "")
        for sid in evidence_source_ids
        if sid and source_provider_by_id.get(sid, "")
    }
    publisher_values = {
        source_publisher_by_id.get(sid, "")
        for sid in evidence_source_ids
        if sid and source_publisher_by_id.get(sid, "")
    }
    evidence_count = max(1, len(evidence_source_ids))
    diversity = clamp(
        (
            (unique_sources / evidence_count)
            + (len(provider_values) / evidence_count)
            + (len(publisher_values) / evidence_count)
        )
        / 3.0,
        0.0,
        1.0,
    )

    verification_quality = 0.0
    if verification_scores:
        verification_quality = clamp(sum(verification_scores) / len(verification_scores), 0.0, 1.0)

    return {
        "strength": round(strength, 6),
        "diversity": round(diversity, 6),
        "verification": round(verification_quality, 6),
        "conflict_penalty": round(clamp(conflict_penalty, 0.0, 1.0), 6),
    }


def score_from_components(components: Dict[str, float]) -> float:
    base = (
        (0.65 * components.get("strength", 0.0))
        + (0.20 * components.get("diversity", 0.0))
        + (0.15 * components.get("verification", 0.0))
    )
    return clamp(base * components.get("conflict_penalty", 1.0), 0.0, 1.0)


def calibrate_confidence_ratings(scores: Sequence[float]) -> List[int]:
    """Quantile calibration fallback when no supervised calibrator is available."""
    if not scores:
        return []
    rounded_unique = {round(float(s), 8) for s in scores}
    if len(rounded_unique) == 1:
        score = clamp(float(scores[0]), 0.0, 1.0)
        # A neutral default for fully tied samples avoids artificial spread.
        baseline = int(round(clamp(1.0 + score * 4.0, 1.0, 5.0)))
        return [baseline for _ in scores]

    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ratings = [1 for _ in scores]
    max_rank = max(1, len(scores) - 1)
    for rank, idx in enumerate(order):
        pct = rank / max_rank
        ratings[idx] = int(clamp(1.0 + (pct * 4.0), 1.0, 5.0))
    return ratings


def compute_claim_confidence(evidence_weights: Iterable[float]) -> Tuple[float, int]:
    """Backward-compatible score+rating path for unit tests and helper callers."""
    score = compute_evidence_strength(evidence_weights)
    rating = int(round(clamp(score * 5.0, 1.0, 5.0)))
    return score, rating
