from agent.orchestrator import _build_meta_by_url


def test_meta_precedence_prefers_academic_over_duckduckgo():
    url = "https://arxiv.org/abs/1234.5678"
    results = [
        {
            "url": url,
            "provider": "duckduckgo",
            "source_type": "other",
            "title": "Short",
        },
        {
            "url": url,
            "provider": "arxiv",
            "source_type": "preprint",
            "title": "Longer academic title",
        },
    ]
    meta_by_url = _build_meta_by_url(results)
    chosen = meta_by_url[url]
    assert chosen["provider"] == "arxiv"
    assert chosen["source_type"] == "preprint"
