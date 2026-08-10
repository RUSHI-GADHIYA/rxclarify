"""openFDA drug-label client.

Responses are cached to data/raw/ keyed by the search term, so re-running the
ingest costs zero API calls. Unauthenticated quota is 240 requests/minute and
1,000/day per IP; a full corpus fetch is ~55 requests, well inside that.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from rxclarify.config import DRUG_LIST_PATH, RAW_DIR, get_settings
from rxclarify.ingest.select import CANDIDATE_LIMIT

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"

# Politeness delay between live calls. The documented ceiling is 240/min; this
# keeps us at ~5/sec with plenty of headroom.
REQUEST_DELAY_SECONDS = 0.2


class OpenFDANotFound(Exception):
    """No prescription label matched the search term."""


@dataclass(frozen=True)
class FetchResult:
    generic: str
    # Several candidates per generic; ingest.select picks the best one. See
    # select.py for why taking openFDA's first hit is not good enough.
    candidates: list[dict]
    from_cache: bool


def load_drug_list(path: Path | None = None) -> list[str]:
    """Flatten data/drug_list.yml (class -> [generic, ...]) into one ordered list."""
    path = path or DRUG_LIST_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    generics: list[str] = []
    seen: set[str] = set()
    for names in raw.values():
        for name in names or []:
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                generics.append(key)
    return generics


def _cache_path(generic: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", generic.lower()).strip("_")
    return RAW_DIR / f"{slug}.json"


def fetch_label(
    generic: str,
    *,
    client: httpx.Client,
    refresh: bool = False,
) -> FetchResult:
    """Fetch one prescription label for `generic`, preferring the on-disk cache.

    Raises OpenFDANotFound when openFDA has no matching prescription label.
    """
    cache = _cache_path(generic)
    if cache.exists() and not refresh:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        # Tolerate the pre-candidate cache format (a single payload object).
        candidates = cached if isinstance(cached, list) else [cached]
        return FetchResult(generic, candidates, True)

    settings = get_settings()
    params: dict[str, str | int] = {
        # Restrict to prescription labels: OTC monographs are structured very
        # differently and mostly lack a drug_interactions section.
        "search": (
            f'openfda.generic_name:"{generic}" AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
        ),
        "limit": CANDIDATE_LIMIT,
    }
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key

    response = client.get(OPENFDA_LABEL_URL, params=params)
    time.sleep(REQUEST_DELAY_SECONDS)

    # openFDA returns 404 with a NOT_FOUND error body for an empty result set.
    if response.status_code == 404:
        raise OpenFDANotFound(generic)
    response.raise_for_status()

    results = response.json().get("results") or []
    if not results:
        raise OpenFDANotFound(generic)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return FetchResult(generic, results, False)


def fetch_all(generics: list[str], *, refresh: bool = False) -> tuple[list[FetchResult], list[str]]:
    """Fetch every generic. Returns (results, names that had no label)."""
    results: list[FetchResult] = []
    missing: list[str] = []
    with httpx.Client(timeout=30.0, headers={"User-Agent": "rxclarify/0.1"}) as client:
        for generic in generics:
            try:
                results.append(fetch_label(generic, client=client, refresh=refresh))
            except OpenFDANotFound:
                missing.append(generic)
    return results, missing
