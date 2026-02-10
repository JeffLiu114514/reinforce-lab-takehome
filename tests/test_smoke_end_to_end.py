import os
import pytest

from agent.orchestrator import run


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_smoke_end_to_end(tmp_path):
    prompt = (
        "What are the real-world risks and benefits of using synthetic data to "
        "train or fine-tune large language models? Focus on data quality, bias, and evaluation."
    )
    ledger = run(prompt=prompt, k_per_query=2, max_urls=5, out_dir=str(tmp_path))
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "ledger.json").exists()
    assert (tmp_path / "trace.json").exists()
    assert (tmp_path / "graph.mmd").exists()
    claim_types = {c.claim_type for c in ledger.graph.claims}
    assert len(claim_types) >= 2
    assert all(c.supported_by for c in ledger.graph.claims)
    assert ledger.metrics is not None
    assert ledger.resolutions is not None
    assert "ece" in ledger.metrics
    assert "brier" in ledger.metrics
    assert "provider_health" in ledger.metrics

