from __future__ import annotations

import json
from pathlib import Path

import time

from fastapi.testclient import TestClient

import agent.webapp as webapp
from agent.schemas import Claim, ClaimGraph, EvidenceCard, Ledger, Source


def _fake_ledger(prompt: str) -> Ledger:
    source = Source(id="S1", url="https://example.com", title="Example source")
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
        statement="Synthetic data can improve coverage.",
        supported_by=["E1"],
        confidence=3,
        confidence_score=0.5,
    )
    return Ledger(
        prompt=prompt,
        plan={"angles": [], "queries": [], "constraints": []},
        sources=[source],
        evidence=[evidence],
        graph=ClaimGraph(claims=[claim], edges=[]),
        created_at="2026-01-01T00:00:00Z",
        metrics={"ece": 0.1, "brier": 0.2},
    )


def test_web_run_endpoint(monkeypatch, tmp_path):
    out_dir = tmp_path / "artifacts"

    def fake_run(prompt, k_per_query, max_urls, out_dir, config_path, progress_hook=None):
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.md").write_text("# Report", encoding="utf-8")
        (output_dir / "graph.mmd").write_text("graph TD", encoding="utf-8")
        (output_dir / "trace.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        (output_dir / "ledger.json").write_text("{}", encoding="utf-8")
        return _fake_ledger(prompt)

    monkeypatch.setattr(webapp, "run_agent", fake_run)

    client = TestClient(webapp.create_app())
    response = client.post(
        "/api/run",
        json={
            "prompt": "test prompt",
            "out_dir": str(out_dir),
            "k_per_query": 2,
            "max_urls": 4,
            "config_path": "config/source_weights.json",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True

    run_id = payload["run_id"]
    status_payload = None
    for _ in range(20):
        status = client.get(f"/api/status/{run_id}")
        status_payload = status.json()
        if status_payload.get("status") == "complete":
            break
        time.sleep(0.01)

    assert status_payload is not None
    assert status_payload["status"] == "complete"
    result = status_payload["result"]
    assert result["summary"]["claims"] == 1
    assert result["report_markdown"] == "# Report"
    assert result["graph_mermaid"] == "graph TD"
    assert result["trace"] == {"ok": True}


def test_web_run_rejects_empty_prompt():
    client = TestClient(webapp.create_app())
    response = client.post("/api/run", json={"prompt": "   "})
    assert response.status_code == 400
