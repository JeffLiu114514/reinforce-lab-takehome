from __future__ import annotations

import math
import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode, urlparse

import httpx
from langchain_community.tools import DuckDuckGoSearchResults

from .utils import normalize_url


def _domain_from_url(url: str) -> str:
    return urlparse(url).netloc.lower()


def _ensure_stats(provider_stats: Dict[str, Dict[str, Any]] | None, provider: str) -> Dict[str, Any]:
    if provider_stats is None:
        return {}
    stats = provider_stats.setdefault(
        provider,
        {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "rate_limited": 0,
            "last_error_code": None,
            "last_error_message": None,
        },
    )
    return stats


def _record_query_provider(
    query_stats: Dict[str, Any] | None,
    provider: str,
    attempted: bool,
    results: int = 0,
    status: str = "ok",
    error_code: str | None = None,
) -> None:
    if query_stats is None:
        return
    query_stats.setdefault("providers", {})[provider] = {
        "attempted": attempted,
        "results": int(results),
        "status": status,
        "error_code": error_code,
    }


def _request_with_retries(
    url: str,
    headers: Dict[str, str] | None = None,
    timeout: float = 20.0,
    max_retries: int = 2,
) -> Tuple[httpx.Response | None, str | None, str | None, bool]:
    last_code: str | None = None
    last_message: str | None = None
    rate_limited = False
    for attempt in range(max_retries + 1):
        try:
            resp = httpx.get(url, timeout=timeout, headers=headers, follow_redirects=True)
            if resp.status_code == 429:
                rate_limited = True
                last_code = "429"
                last_message = "rate_limited"
                if attempt < max_retries:
                    time.sleep(0.4 * (2**attempt))
                    continue
            resp.raise_for_status()
            return resp, None, None, rate_limited
        except httpx.HTTPStatusError as exc:
            code = str(exc.response.status_code) if exc.response is not None else "http_error"
            last_code = code
            last_message = str(exc)
            if code in {"429", "500", "502", "503", "504"} and attempt < max_retries:
                time.sleep(0.4 * (2**attempt))
                continue
            break
        except Exception as exc:  # pragma: no cover - network transport variance
            last_code = "request_error"
            last_message = str(exc)
            if attempt < max_retries:
                time.sleep(0.4 * (2**attempt))
                continue
            break
    return None, last_code, last_message, rate_limited


def search_duckduckgo(
    query: str,
    k: int = 5,
    provider_stats: Dict[str, Dict[str, Any]] | None = None,
    query_stats: Dict[str, Any] | None = None,
) -> List[Dict]:
    stats = _ensure_stats(provider_stats, "duckduckgo")
    if stats:
        stats["attempts"] += 1
    tool = DuckDuckGoSearchResults(max_results=k, output_format="list")
    try:
        results = tool.invoke(query)
    except Exception as exc:  # pragma: no cover - provider behavior varies
        if stats:
            stats["failures"] += 1
            stats["last_error_code"] = "provider_error"
            stats["last_error_message"] = str(exc)
        _record_query_provider(query_stats, "duckduckgo", attempted=True, results=0, status="error", error_code="provider_error")
        return []
    if isinstance(results, str):
        if stats:
            stats["failures"] += 1
            stats["last_error_code"] = "invalid_payload"
            stats["last_error_message"] = "provider returned string payload"
        _record_query_provider(query_stats, "duckduckgo", attempted=True, results=0, status="error", error_code="invalid_payload")
        return []
    cleaned = []
    for item in results:
        url = item.get("link") or item.get("url") or ""
        if not url:
            continue
        cleaned.append(
            {
                "url": normalize_url(url),
                "title": item.get("title") or "",
                "snippet": item.get("snippet") or item.get("body") or "",
                "provider": "duckduckgo",
                "source_type": "other",
                "publisher": _domain_from_url(url),
                "provider_status": "ok",
                "provider_error_code": None,
            }
        )
    if stats:
        if cleaned:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
            stats["last_error_code"] = "empty_results"
            stats["last_error_message"] = "no results returned"
    _record_query_provider(
        query_stats,
        "duckduckgo",
        attempted=True,
        results=len(cleaned),
        status="ok" if cleaned else "empty",
        error_code=None if cleaned else "empty_results",
    )
    return cleaned


def search_semantic_scholar(
    query: str,
    k: int = 5,
    provider_stats: Dict[str, Dict[str, Any]] | None = None,
    query_stats: Dict[str, Any] | None = None,
) -> List[Dict]:
    stats = _ensure_stats(provider_stats, "semantic_scholar")
    if stats:
        stats["attempts"] += 1
    params = {
        "query": query,
        "limit": k,
        "fields": "title,abstract,authors,venue,year,url,publicationTypes,externalIds",
    }
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(params)
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    resp, err_code, err_message, rate_limited = _request_with_retries(
        url=url,
        headers=headers or None,
        max_retries=2,
    )
    if resp is None:
        if stats:
            stats["failures"] += 1
            stats["last_error_code"] = err_code
            stats["last_error_message"] = err_message
            if rate_limited:
                stats["rate_limited"] += 1
        _record_query_provider(
            query_stats,
            "semantic_scholar",
            attempted=True,
            results=0,
            status="error",
            error_code=err_code,
        )
        return []
    if stats and rate_limited:
        stats["rate_limited"] += 1
    data = resp.json() if resp.text else {}
    papers = data.get("data", [])
    results = []
    for paper in papers:
        title = paper.get("title") or ""
        snippet = paper.get("abstract") or ""
        url_val = paper.get("url") or ""
        external = paper.get("externalIds") or {}
        if not url_val and external.get("ArXiv"):
            url_val = f"https://arxiv.org/abs/{external.get('ArXiv')}"
        if not url_val:
            continue
        pub_types = paper.get("publicationTypes") or []
        source_type = "paper" if "JournalArticle" in pub_types else "preprint" if "Preprint" in pub_types else "report"
        results.append(
            {
                "url": normalize_url(url_val),
                "title": title,
                "snippet": snippet,
                "provider": "semantic_scholar",
                "source_type": source_type,
                "publisher": paper.get("venue") or "Semantic Scholar",
                "provider_status": "ok",
                "provider_error_code": None,
            }
        )
    if stats:
        if results:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
            stats["last_error_code"] = "empty_results"
            stats["last_error_message"] = "no papers returned"
    _record_query_provider(
        query_stats,
        "semantic_scholar",
        attempted=True,
        results=len(results),
        status="ok" if results else "empty",
        error_code=None if results else "empty_results",
    )
    return results


def search_arxiv(
    query: str,
    k: int = 5,
    provider_stats: Dict[str, Dict[str, Any]] | None = None,
    query_stats: Dict[str, Any] | None = None,
) -> List[Dict]:
    stats = _ensure_stats(provider_stats, "arxiv")
    if stats:
        stats["attempts"] += 1
    url = (
        "https://export.arxiv.org/api/query?"
        + urlencode({"search_query": f"all:{query}", "start": 0, "max_results": k})
    )
    resp, err_code, err_message, _ = _request_with_retries(url=url, max_retries=1)
    if resp is None:
        if stats:
            stats["failures"] += 1
            stats["last_error_code"] = err_code
            stats["last_error_message"] = err_message
        _record_query_provider(
            query_stats,
            "arxiv",
            attempted=True,
            results=0,
            status="error",
            error_code=err_code,
        )
        return []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        if stats:
            stats["failures"] += 1
            stats["last_error_code"] = "parse_error"
            stats["last_error_message"] = "invalid atom payload"
        _record_query_provider(
            query_stats,
            "arxiv",
            attempted=True,
            results=0,
            status="error",
            error_code="parse_error",
        )
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        link = ""
        for link_el in entry.findall("atom:link", ns):
            if link_el.attrib.get("rel") == "alternate":
                link = link_el.attrib.get("href", "")
                break
        if not link:
            link = entry.findtext("atom:id", default="", namespaces=ns) or ""
        if not link:
            continue
        results.append(
            {
                "url": normalize_url(link),
                "title": title,
                "snippet": summary,
                "provider": "arxiv",
                "source_type": "preprint",
                "publisher": "arXiv",
                "provider_status": "ok",
                "provider_error_code": None,
            }
        )
    if stats:
        if results:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
            stats["last_error_code"] = "empty_results"
            stats["last_error_message"] = "no entries returned"
    _record_query_provider(
        query_stats,
        "arxiv",
        attempted=True,
        results=len(results),
        status="ok" if results else "empty",
        error_code=None if results else "empty_results",
    )
    return results


def route_search(
    query_spec: Dict,
    k: int = 5,
    provider_stats: Dict[str, Dict[str, Any]] | None = None,
    return_stats: bool = False,
) -> List[Dict] | Tuple[List[Dict], Dict[str, Any]]:
    hint = query_spec.get("provider_hint", "general")
    results: List[Dict] = []
    query_stats: Dict[str, Any] = {
        "query": query_spec.get("q", ""),
        "provider_hint": hint,
        "providers": {},
    }
    if hint == "academic":
        k1 = max(1, k // 2)
        k2 = max(1, k - k1)
        results.extend(
            search_semantic_scholar(
                query_spec["q"],
                k1,
                provider_stats=provider_stats,
                query_stats=query_stats,
            )
        )
        results.extend(
            search_arxiv(
                query_spec["q"],
                k2,
                provider_stats=provider_stats,
                query_stats=query_stats,
            )
        )
    elif hint == "industry":
        results.extend(
            search_duckduckgo(
                query_spec["q"],
                k,
                provider_stats=provider_stats,
                query_stats=query_stats,
            )
        )
    else:
        k_web = max(1, int(math.ceil(k * 0.6)))
        k_academic = max(1, k - k_web)
        results.extend(
            search_duckduckgo(
                query_spec["q"],
                k_web,
                provider_stats=provider_stats,
                query_stats=query_stats,
            )
        )
        results.extend(
            search_semantic_scholar(
                query_spec["q"],
                max(1, k_academic // 2),
                provider_stats=provider_stats,
                query_stats=query_stats,
            )
        )
        results.extend(
            search_arxiv(
                query_spec["q"],
                max(1, k_academic - max(1, k_academic // 2)),
                provider_stats=provider_stats,
                query_stats=query_stats,
            )
        )

    seen = set()
    deduped = []
    for item in results:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(item)
    if return_stats:
        query_stats["result_count"] = len(deduped)
        return deduped, query_stats
    return deduped
