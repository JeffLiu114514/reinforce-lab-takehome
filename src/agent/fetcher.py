from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from diskcache import Cache
import trafilatura

from .schemas import Source
from .utils import hash_text, normalize_url, sanitize_whitespace


class Fetcher:
    def __init__(self, cache_dir: str = ".cache") -> None:
        self.cache = Cache(cache_dir)
        self.client = httpx.Client(
            timeout=20.0,
            headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()
        self.cache.close()

    def _extract_text(self, html: str) -> str:
        extracted = trafilatura.extract(html) or ""
        if len(extracted.strip()) >= 1000:
            return sanitize_whitespace(extracted)
        # fallback to BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ")
        return sanitize_whitespace(text)

    def fetch_one(self, url: str) -> Optional[Tuple[Dict, str]]:
        canonical = normalize_url(url)
        cache_key = f"page::{hash_text(canonical)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        try:
            resp = self.client.get(canonical)
            resp.raise_for_status()
        except Exception:
            return None
        html = resp.text
        text = self._extract_text(html)
        data = ({"url": canonical, "title": self._guess_title(html)}, text)
        self.cache.set(cache_key, data)
        return data

    def _guess_title(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string if soup.title and soup.title.string else ""
        return sanitize_whitespace(title)


def build_source(source_id: str, url: str, title: str) -> Source:
    return Source(id=source_id, url=url, title=title or url)


def fetch_sources(urls: List[str], cache_dir: str = ".cache") -> List[Tuple[Source, str]]:
    fetcher = Fetcher(cache_dir=cache_dir)
    sources_with_text: List[Tuple[Source, str]] = []
    source_counter = 0
    try:
        for url in urls:
            result = fetcher.fetch_one(url)
            if not result:
                continue
            meta, text = result
            source_counter += 1
            source = build_source(
                f"S{source_counter}",
                meta.get("url", url),
                meta.get("title", ""),
            )
            sources_with_text.append((source, text))
    finally:
        fetcher.close()
    return sources_with_text

