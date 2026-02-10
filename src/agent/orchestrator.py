from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

from .claim_builder import build_claims
from .claim_cluster import canonicalize_claims
from .config import load_config
from .contradiction import build_edges
from .extractor import extract_evidence
from .fetcher import fetch_sources
from .mermaid import render_claim_graph
from .metrics import compute_metrics
from .planner import plan_research
from .report import render_report
from .resolver import resolve_contradictions
from .schemas import ClaimGraph, EvidenceCard, Ledger, Source
from .search_providers import route_search
from .utils import IdGenerator, get_domain, hash_text, normalize_url, now_utc_iso
from .weights import (
    calibrate_confidence_ratings,
    compute_claim_confidence_components,
    compute_evidence_strength,
    compute_evidence_weight,
    compute_source_weight,
    score_from_components,
)


PROVIDER_PRIORITY = {
    "semantic_scholar": 3,
    "arxiv": 2,
    "duckduckgo": 1,
}

SOURCE_TYPE_PRIORITY = {
    "paper": 5,
    "preprint": 4,
    "report": 3,
    "documentation": 2,
    "news": 2,
    "blog": 1,
    "other": 0,
}


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for url in urls:
        canonical = normalize_url(url)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def _pick_meta(existing: Dict[str, Any] | None, candidate: Dict[str, Any]) -> Dict[str, Any]:
    if existing is None:
        return candidate

    existing_provider_score = PROVIDER_PRIORITY.get(existing.get("provider"), 0)
    candidate_provider_score = PROVIDER_PRIORITY.get(candidate.get("provider"), 0)
    if candidate_provider_score > existing_provider_score:
        return candidate
    if candidate_provider_score < existing_provider_score:
        return existing

    existing_type_score = SOURCE_TYPE_PRIORITY.get(existing.get("source_type", "other"), 0)
    candidate_type_score = SOURCE_TYPE_PRIORITY.get(candidate.get("source_type", "other"), 0)
    if candidate_type_score > existing_type_score:
        return candidate
    if candidate_type_score < existing_type_score:
        return existing

    existing_title_len = len(existing.get("title") or "")
    candidate_title_len = len(candidate.get("title") or "")
    if candidate_title_len > existing_title_len:
        return candidate
    return existing


def _build_meta_by_url(results_all: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    meta_by_url: Dict[str, Dict[str, Any]] = {}
    for result in results_all:
        url = normalize_url(result.get("url", ""))
        if not url:
            continue
        current = meta_by_url.get(url)
        meta_by_url[url] = _pick_meta(current, result)
    return meta_by_url


def _infer_source_type(url: str, source_type: str | None) -> str:
    if source_type and source_type != "other":
        return source_type
    domain = get_domain(url)
    lowered = url.lower()
    if "arxiv.org" in domain:
        return "preprint"
    if any(token in domain for token in ("medium.com", "substack.com", "blog")):
        return "blog"
    if any(token in domain for token in ("news", "nytimes", "reuters", "bbc", "cnn")):
        return "news"
    if any(token in lowered for token in ("/docs/", "/documentation/", "/manual/")):
        return "documentation"
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return "report"
    return "other"


def _verification_quality(card: EvidenceCard) -> float:
    if not card.verified:
        return 0.0
    if card.verification_method == "exact":
        return 1.0
    if card.verification_method == "fuzzy":
        score = card.verification_score or 0.0
        return max(0.0, min(0.95, float(score) / 100.0))
    return 0.0


def _conflict_penalty_by_claim(claim_ids: List[str], edges: List, resolutions: List) -> Dict[str, float]:
    penalty = {cid: 1.0 for cid in claim_ids}
    resolution_by_id = {r.id: r for r in resolutions}
    for edge in edges:
        if edge.relation != "contradicts":
            continue
        for cid in (edge.src_claim_id, edge.dst_claim_id):
            edge_penalty = 0.8
            if edge.resolution_id and edge.resolution_id in resolution_by_id:
                res = resolution_by_id[edge.resolution_id]
                if res.leaning_claim_id == cid:
                    edge_penalty = 0.95
                elif res.leaning_claim_id and res.leaning_claim_id != cid:
                    edge_penalty = 0.68
                else:
                    edge_penalty = 0.78
            penalty[cid] = min(penalty.get(cid, 1.0), edge_penalty)
    return penalty


def _log(message: str, hook: Callable[[str], None] | None) -> None:
    print(message)
    if hook:
        hook(message)


def run(
    prompt: str,
    k_per_query: int = 8,
    max_urls: int = 30,
    out_dir: str = "artifacts",
    config_path: str = "config/source_weights.json",
    progress_hook: Callable[[str], None] | None = None,
) -> Ledger:
    config = load_config(config_path)
    _log("[plan] generating research plan", progress_hook)
    plan = plan_research(prompt)

    _log("[search] running queries", progress_hook)
    urls: List[str] = []
    results_all: List[Dict] = []
    provider_stats: Dict[str, Dict[str, Any]] = {}
    query_runs: List[Dict[str, Any]] = []
    for q in plan.queries:
        results, query_stats = route_search(
            q.model_dump(),
            k=k_per_query,
            provider_stats=provider_stats,
            return_stats=True,
        )
        results_all.extend(results)
        query_runs.append(query_stats)
        urls.extend([r["url"] for r in results])
    urls = _dedupe_urls(urls)[:max_urls]
    _log(f"[search] candidate urls: {len(urls)}", progress_hook)
    meta_by_url = _build_meta_by_url(results_all)

    _log("[fetch] fetching and extracting sources", progress_hook)
    sources_with_text = fetch_sources(urls, cache_dir=os.path.join(out_dir, ".cache"))
    sources: List[Source] = [s for s, _ in sources_with_text]
    for source in sources:
        meta = meta_by_url.get(normalize_url(str(source.url)))
        provider = meta.get("provider") if meta else None
        source_type = _infer_source_type(str(source.url), meta.get("source_type") if meta else None)
        publisher = meta.get("publisher") if meta else None
        domain = get_domain(str(source.url))
        source.provider = provider
        source.provider_status = meta.get("provider_status") if meta else None
        source.provider_error_code = meta.get("provider_error_code") if meta else None
        source.source_type = source_type
        if publisher:
            source.publisher = publisher
        source.domain = domain
        source.source_weight = compute_source_weight(
            provider=provider,
            source_type=source.source_type,
            domain=domain,
            config=config,
        )
    _log(f"[fetch] extracted sources: {len(sources)}", progress_hook)

    _log("[evidence] extracting evidence snippets", progress_hook)
    evidence_cards: List[EvidenceCard] = []
    evidence_id_gen = IdGenerator(prefix="E")
    seen_snippets = set()
    verification_cfg = config.get("verification", {})
    for source, text in sources_with_text:
        if not text:
            continue
        for card in extract_evidence(prompt, source, text, verification_cfg=verification_cfg):
            snippet_hash = hash_text(card.snippet)
            if snippet_hash in seen_snippets:
                continue
            seen_snippets.add(snippet_hash)
            card.id = evidence_id_gen.next()
            # Down-weight thin pages that often include navigation noise.
            if len(text) < 1000 and card.reliability > 2:
                card.reliability = 2
            card.evidence_weight = compute_evidence_weight(
                source_weight=source.source_weight,
                reliability=card.reliability,
            )
            evidence_cards.append(card)
    _log(f"[evidence] cards: {len(evidence_cards)}", progress_hook)

    _log("[claims] building claims", progress_hook)
    claims = build_claims(evidence_cards)
    evidence_by_id = {e.id: e for e in evidence_cards}
    for claim in claims:
        weights = [evidence_by_id[eid].evidence_weight for eid in claim.supported_by if eid in evidence_by_id]
        claim.confidence_score = compute_evidence_strength(weights)
    similarity_threshold = float(config.get("clustering", {}).get("similarity_threshold", 0.8))
    claims = canonicalize_claims(claims, similarity_threshold, {e.id: e.evidence_weight for e in evidence_cards})
    _log(f"[claims] total: {len(claims)}", progress_hook)

    _log("[graph] building contradiction edges", progress_hook)
    edges = build_edges(claims)
    _log(f"[graph] edges: {len(edges)}", progress_hook)

    _log("[resolve] resolving contradictions", progress_hook)
    resolutions, edges = resolve_contradictions(claims, edges, evidence_by_id)
    _log(f"[resolve] resolutions: {len(resolutions)}", progress_hook)

    source_provider_by_id = {s.id: (s.provider or "") for s in sources}
    source_publisher_by_id = {s.id: (s.publisher or "") for s in sources}
    conflict_penalty = _conflict_penalty_by_claim([c.id for c in claims], edges, resolutions)

    for claim in claims:
        supporting = [evidence_by_id[eid] for eid in claim.supported_by if eid in evidence_by_id]
        components = compute_claim_confidence_components(
            evidence_weights=[ev.evidence_weight for ev in supporting],
            evidence_source_ids=[ev.source_id for ev in supporting],
            source_provider_by_id=source_provider_by_id,
            source_publisher_by_id=source_publisher_by_id,
            verification_scores=[_verification_quality(ev) for ev in supporting],
            conflict_penalty=conflict_penalty.get(claim.id, 1.0),
        )
        claim.confidence_components = components
        claim.confidence_score = score_from_components(components)
        claim.needs_more_evidence = (
            components.get("strength", 0.0) < 0.25
            or components.get("diversity", 0.0) < 0.45
            or components.get("conflict_penalty", 1.0) < 0.8
        )

    calibrated = calibrate_confidence_ratings([c.confidence_score for c in claims])
    for claim, rating in zip(claims, calibrated):
        claim.confidence = rating

    metrics = compute_metrics(claims, edges, evidence_cards, sources, provider_stats=provider_stats)

    ledger = Ledger(
        prompt=prompt,
        plan=plan.model_dump(),
        sources=sources,
        evidence=evidence_cards,
        graph=ClaimGraph(claims=claims, edges=edges),
        resolutions=resolutions,
        metrics=metrics,
        created_at=now_utc_iso(),
    )

    os.makedirs(out_dir, exist_ok=True)
    ledger_path = os.path.join(out_dir, "ledger.json")
    report_path = os.path.join(out_dir, "report.md")
    trace_path = os.path.join(out_dir, "trace.json")
    graph_path = os.path.join(out_dir, "graph.mmd")

    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger.model_dump(mode="json"), f, indent=2)

    report_md = render_report(ledger)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(graph_path, "w", encoding="utf-8") as f:
        f.write(render_claim_graph(claims, edges))

    trace = {
        "prompt": prompt,
        "plan": plan.model_dump(),
        "queries": [q.model_dump() for q in plan.queries],
        "query_runs": query_runs,
        "provider_stats": provider_stats,
        "urls": urls,
        "sources": [s.model_dump(mode="json") for s in sources],
        "evidence": [e.model_dump(mode="json") for e in evidence_cards],
        "claims": [c.model_dump(mode="json") for c in claims],
        "edges": [e.model_dump(mode="json") for e in edges],
        "resolutions": [r.model_dump(mode="json") for r in resolutions],
        "metrics": metrics,
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)

    _log(f"[done] wrote {ledger_path} and {report_path}", progress_hook)
    return ledger
