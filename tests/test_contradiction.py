from agent.contradiction import build_edges
from agent.schemas import Claim


def test_contradiction_heuristic():
    claims = [
        Claim(
            id="C1",
            claim_type="evaluation",
            statement="Synthetic data improves generalization.",
            polarity="pro",
            supported_by=["E1"],
            confidence=3,
        ),
        Claim(
            id="C2",
            claim_type="evaluation",
            statement="Synthetic data harms generalization in real deployment.",
            polarity="con",
            supported_by=["E2"],
            confidence=3,
        ),
        Claim(
            id="C3",
            claim_type="evaluation",
            statement="Synthetic data helps when distribution matches target.",
            polarity="mixed",
            supported_by=["E3"],
            confidence=3,
        ),
    ]
    edges = build_edges(claims, llm=None)
    assert any(e.relation in {"contradicts", "refines"} for e in edges)

