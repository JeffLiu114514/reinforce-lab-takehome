from agent.schemas import Claim, ClaimGraph, EvidenceCard, Ledger, Source


def test_schemas_roundtrip():
    source = Source(id="S1", url="https://example.com", title="Example")
    evidence = EvidenceCard(
        id="E1",
        source_id="S1",
        claim_types=["data_quality"],
        snippet="Synthetic data can improve coverage.",
        reliability=3,
    )
    claim = Claim(
        id="C1",
        claim_type="data_quality",
        statement="Synthetic data can improve data coverage.",
        supported_by=["E1"],
        confidence=3,
    )
    graph = ClaimGraph(claims=[claim], edges=[])
    ledger = Ledger(
        prompt="test",
        plan={"angles": [], "queries": []},
        sources=[source],
        evidence=[evidence],
        graph=graph,
        created_at="2020-01-01T00:00:00Z",
    )
    assert ledger.graph.claims[0].supported_by == ["E1"]

