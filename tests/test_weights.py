from agent.config import load_config
from agent.weights import (
    calibrate_confidence_ratings,
    compute_claim_confidence,
    compute_claim_confidence_components,
    compute_evidence_weight,
    compute_source_weight,
    score_from_components,
)


def test_weight_calculations():
    cfg = load_config(None)
    source_weight = compute_source_weight("semantic_scholar", "paper", "arxiv.org", cfg)
    assert 0.1 <= source_weight <= 1.0

    evidence_weight = compute_evidence_weight(source_weight, reliability=4)
    assert 0.0 <= evidence_weight <= 1.0

    score, rating = compute_claim_confidence([evidence_weight, 0.2])
    assert 0.0 <= score <= 1.0
    assert 1 <= rating <= 5


def test_claim_confidence_components():
    components = compute_claim_confidence_components(
        evidence_weights=[0.6, 0.4],
        evidence_source_ids=["S1", "S2"],
        source_provider_by_id={"S1": "semantic_scholar", "S2": "arxiv"},
        source_publisher_by_id={"S1": "Nature", "S2": "arXiv"},
        verification_scores=[1.0, 0.9],
        conflict_penalty=0.95,
    )
    assert components["strength"] > 0.0
    assert components["diversity"] > 0.0
    assert components["verification"] > 0.0
    assert components["conflict_penalty"] == 0.95
    score = score_from_components(components)
    assert 0.0 <= score <= 1.0


def test_quantile_calibration_spreads_non_tied_scores():
    ratings = calibrate_confidence_ratings([0.1, 0.3, 0.5, 0.9])
    assert len(set(ratings)) >= 3
