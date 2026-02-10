from agent.resolver import resolve_contradictions
from agent.schemas import Claim, Edge, EvidenceCard


def test_resolution_component_weights():
    evidence = {
        "E1": EvidenceCard(
            id="E1",
            source_id="S1",
            claim_types=["evaluation"],
            snippet="A",
            reliability=4,
            evidence_weight=0.8,
        ),
        "E2": EvidenceCard(
            id="E2",
            source_id="S1",
            claim_types=["evaluation"],
            snippet="B",
            reliability=3,
            evidence_weight=0.4,
        ),
    }
    claims = [
        Claim(
            id="C1",
            claim_type="evaluation",
            statement="Synthetic data improves generalization.",
            supported_by=["E1"],
            confidence=3,
            confidence_score=0.8,
        ),
        Claim(
            id="C2",
            claim_type="evaluation",
            statement="Synthetic data harms generalization.",
            supported_by=["E2"],
            confidence=3,
            confidence_score=0.4,
        ),
    ]
    edges = [
        Edge(
            src_claim_id="C1",
            dst_claim_id="C2",
            relation="contradicts",
        )
    ]
    resolutions, updated_edges = resolve_contradictions(claims, edges, evidence, llm=None)
    assert len(resolutions) == 1
    assert resolutions[0].weight_by_claim["C1"] > resolutions[0].weight_by_claim["C2"]
    assert updated_edges[0].resolution_id == resolutions[0].id
