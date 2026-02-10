from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import Claim
from .weights import compute_claim_confidence


def _dedupe_list(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def canonicalize_claims(
    claims: List[Claim],
    similarity_threshold: float,
    evidence_weights: Dict[str, float],
) -> List[Claim]:
    if len(claims) <= 1:
        return claims

    by_type: Dict[str, List[Claim]] = defaultdict(list)
    for claim in claims:
        by_type[claim.claim_type].append(claim)

    canonical: List[Claim] = []
    for claim_type, group in by_type.items():
        if len(group) == 1:
            canonical.extend(group)
            continue

        texts = [c.statement for c in group]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        sim = cosine_similarity(tfidf)

        parent = list(range(len(group)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if sim[i, j] >= similarity_threshold:
                    union(i, j)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(group)):
            clusters[find(i)].append(i)

        for _, idxs in clusters.items():
            if len(idxs) == 1:
                canonical.append(group[idxs[0]])
                continue

            cluster_claims = [group[i] for i in idxs]
            cluster_claims.sort(key=lambda c: c.confidence_score, reverse=True)
            base = cluster_claims[0]

            merged_supported = _dedupe_list(
                [eid for c in cluster_claims for eid in c.supported_by]
            )
            merged_aliases = _dedupe_list(
                [c.statement for c in cluster_claims[1:]]
                + [alias for c in cluster_claims for alias in c.aliases]
            )

            base.supported_by = merged_supported
            base.aliases = merged_aliases

            weights = [evidence_weights.get(eid, 0.0) for eid in base.supported_by]
            score, rating = compute_claim_confidence(weights)
            base.confidence_score = score
            base.confidence = rating

            canonical.append(base)

    return canonical
