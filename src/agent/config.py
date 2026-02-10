from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "provider_weights": {
        "semantic_scholar": 1.0,
        "arxiv": 0.9,
        "duckduckgo": 0.7,
    },
    "source_type_weights": {
        "paper": 1.0,
        "preprint": 0.9,
        "report": 0.8,
        "blog": 0.6,
        "news": 0.6,
        "other": 0.5,
    },
    "domain_overrides": {
        "arxiv.org": 0.9,
        "ibm.com": 0.8,
    },
    "verification": {
        "fuzzy_threshold": 85,
        "keep_unverified": False,
    },
    "clustering": {
        "similarity_threshold": 0.8,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not path:
        return cfg
    if not os.path.exists(path):
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        override = json.load(f)
    if isinstance(override, dict):
        _deep_merge(cfg, override)
    return cfg
