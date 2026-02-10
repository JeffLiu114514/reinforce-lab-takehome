from __future__ import annotations

from typing import Dict, List

from langchain_community.tools import DuckDuckGoSearchResults

from .utils import normalize_url


def search(query: str, k: int = 5) -> List[Dict]:
    tool = DuckDuckGoSearchResults(max_results=k, output_format="list")
    results = tool.invoke(query)

    if isinstance(results, str):
        return []

    seen = set()
    cleaned = []
    for item in results:
        url = item.get("link") or item.get("url") or ""
        title = item.get("title") or ""
        snippet = item.get("snippet") or item.get("body") or ""
        if not url:
            continue
        canonical = normalize_url(url)
        if canonical in seen:
            continue
        seen.add(canonical)
        cleaned.append({"url": canonical, "title": title, "snippet": snippet})
    return cleaned

