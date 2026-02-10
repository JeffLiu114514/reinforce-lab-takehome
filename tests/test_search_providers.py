import httpx

from agent import search_providers


def test_semantic_scholar_retries_after_rate_limit(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, timeout=None, headers=None, follow_redirects=None):
        calls["count"] += 1
        req = httpx.Request("GET", url)
        if calls["count"] == 1:
            return httpx.Response(429, request=req, json={"message": "too many requests"})
        return httpx.Response(
            200,
            request=req,
            json={
                "data": [
                    {
                        "title": "Paper A",
                        "abstract": "A",
                        "url": "https://example.com/paper-a",
                        "publicationTypes": ["JournalArticle"],
                    }
                ]
            },
        )

    monkeypatch.setattr(search_providers.httpx, "get", fake_get)
    monkeypatch.setattr(search_providers.time, "sleep", lambda *_: None)

    provider_stats = {}
    out = search_providers.search_semantic_scholar(
        "synthetic data",
        k=1,
        provider_stats=provider_stats,
    )
    assert len(out) == 1
    assert provider_stats["semantic_scholar"]["rate_limited"] == 1


def test_arxiv_request_follows_redirects(monkeypatch):
    called = {"follow_redirects": None}
    xml_payload = """<?xml version='1.0' encoding='UTF-8'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry>
        <id>https://arxiv.org/abs/1234.5678</id>
        <title>Example title</title>
        <summary>Example summary</summary>
      </entry>
    </feed>"""

    def fake_get(url, timeout=None, headers=None, follow_redirects=None):
        called["follow_redirects"] = follow_redirects
        req = httpx.Request("GET", url)
        return httpx.Response(200, request=req, text=xml_payload)

    monkeypatch.setattr(search_providers.httpx, "get", fake_get)
    out = search_providers.search_arxiv("synthetic data", k=1, provider_stats={})
    assert len(out) == 1
    assert called["follow_redirects"] is True
