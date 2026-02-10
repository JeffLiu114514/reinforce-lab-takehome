from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k not in TRACKING_PARAMS
    ]
    new_query = urlencode(query, doseq=True)
    cleaned = parsed._replace(query=new_query, fragment="")
    return urlunparse(cleaned)


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(
    text: str, chunk_size: int = 1500, overlap: int = 200, max_chunks: int = 20
) -> List[str]:
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 2)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if max_chunks and len(chunks) >= max_chunks:
            break
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks


def sanitize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class IdGenerator:
    prefix: str
    counter: int = 0

    def next(self) -> str:
        self.counter += 1
        return f"{self.prefix}{self.counter}"


def get_llm() -> ChatGoogleGenerativeAI:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def truncate_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out

