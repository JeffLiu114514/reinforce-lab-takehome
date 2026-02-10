from agent.mermaid import render_claim_graph
from agent.schemas import Claim, Edge


def test_mermaid_keeps_full_claim_text_and_styles():
    statement = (
        "Synthetic data can improve coverage in low-resource settings, "
        "but quality controls are required to avoid drift."
    )
    claims = [
        Claim(
            id="C1",
            claim_type="data_quality",
            statement=statement,
            supported_by=["E1"],
            confidence=4,
            confidence_score=0.7,
        ),
        Claim(
            id="C2",
            claim_type="evaluation",
            statement="Evaluation metrics can miss real-world failures.",
            supported_by=["E2"],
            confidence=2,
            confidence_score=0.3,
        ),
    ]
    edges = [Edge(src_claim_id="C1", dst_claim_id="C2", relation="contradicts")]
    output = render_claim_graph(claims, edges)
    assert "low-resource settings" in output
    assert "style C1 stroke:" in output
    assert "linkStyle 0 stroke:#c62828" in output
