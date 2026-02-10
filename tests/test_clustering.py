from agent.claim_cluster import canonicalize_claims
from agent.schemas import Claim


def test_canonicalize_claims_merges_similar():
    claims = [
        Claim(
            id="C1",
            claim_type="data_quality",
            statement="Synthetic data improves coverage.",
            supported_by=["E1"],
            confidence=3,
            confidence_score=0.6,
        ),
        Claim(
            id="C2",
            claim_type="data_quality",
            statement="Synthetic data improves data coverage.",
            supported_by=["E2"],
            confidence=3,
            confidence_score=0.4,
        ),
    ]
    evidence_weights = {"E1": 0.6, "E2": 0.4}
    out = canonicalize_claims(claims, similarity_threshold=0.4, evidence_weights=evidence_weights)
    assert len(out) == 1
    assert out[0].aliases
