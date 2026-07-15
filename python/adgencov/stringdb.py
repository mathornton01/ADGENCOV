"""STRING-db interaction lookups for the dashboard's heatmap → interactions flow.

When a user highlights a heatmap cell, they pick out a pair of genes; we then ask
STRING-db (https://string-db.org) for the experimentally / curated evidence that
those genes' products interact, plus each gene's strongest interaction partners.

STRING's REST API is free and needs no key.  We use the ``interaction_partners``
endpoint (top partners for each queried protein) in a single POST, and derive the
*direct* pair score from whether each gene appears among the other's partners.

Like :mod:`adgencov.bioquery`, all network access is injectable (``fetch_post``)
so tests run offline, and answers are memoized in :mod:`adgencov.biocache`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .bioquery import BioQueryError, PostFetch, default_post_fetch

__all__ = ["StringError", "Interaction", "interactions", "STRING_CHANNELS"]

_STRING = "https://string-db.org/api"
# STRING identifies the caller for courtesy rate-limiting; not authentication.
_CALLER = "adgencov.thorntonstatistical.com"

# Human-readable names for STRING's per-channel evidence subscores (0–1 each).
STRING_CHANNELS = {
    "nscore": "gene neighborhood",
    "fscore": "gene fusion",
    "pscore": "phylogenetic co-occurrence",
    "ascore": "coexpression",
    "escore": "experimental",
    "dscore": "database (curated)",
    "tscore": "text mining",
}

# Common organism aliases → NCBI taxon id (STRING's ``species`` param).
_SPECIES_TAXID = {
    "human": 9606, "homo sapiens": 9606, "9606": 9606,
    "mouse": 10090, "mus musculus": 10090, "10090": 10090,
    "rat": 10116, "rattus norvegicus": 10116, "10116": 10116,
    "worm": 6239, "celegans": 6239, "c. elegans": 6239, "c elegans": 6239,
    "caenorhabditis elegans": 6239, "nematode": 6239, "6239": 6239,
}


class StringError(BioQueryError):
    """Raised when a STRING-db lookup cannot be performed or parsed."""


@dataclass
class Interaction:
    """One STRING edge: a queried gene and one of its interaction partners."""

    query: str          # the queried gene (preferredName_A)
    partner: str        # the interacting partner (preferredName_B)
    score: float        # combined STRING score, 0–1
    channels: Dict[str, float]  # per-evidence subscores, 0–1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "partner": self.partner,
            "score": self.score,
            "channels": self.channels,
        }


def _taxid(species: Any) -> int:
    s = str(species or "9606").strip().lower()
    if s in _SPECIES_TAXID:
        return _SPECIES_TAXID[s]
    try:
        return int(s)
    except ValueError:
        raise StringError(f"unknown species {species!r}")


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _direct_score(fetch_post: PostFetch, genes: List[str], tax: int) -> Optional[float]:
    """Strongest STRING edge *among* the queried genes (the direct pair link)."""
    fields = {
        "identifiers": "\r".join(genes),
        "species": str(tax),
        "caller_identity": _CALLER,
    }
    try:
        raw = fetch_post(f"{_STRING}/json/network", fields)
        rows = json.loads(raw)
    except (BioQueryError, ValueError, TypeError):
        return None
    if not isinstance(rows, list):
        return None
    wanted = {g.upper() for g in genes}
    best: Optional[float] = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        a = str(row.get("preferredName_A", "")).strip().upper()
        b = str(row.get("preferredName_B", "")).strip().upper()
        if a in wanted and b in wanted and a != b:
            s = _num(row.get("score"))
            best = s if best is None else max(best, s)
    return best


def interactions(
    genes: List[str],
    *,
    species: Any = 9606,
    limit: int = 10,
    required_score: int = 400,
    fetch_post: Optional[PostFetch] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Look up STRING interaction partners for one or two genes.

    Parameters
    ----------
    genes : list of str
        One or two gene symbols (a heatmap cell picks two; the diagonal, one).
    species : int or str
        NCBI taxon id or an alias (``human``/``mouse``/``rat``).  Default human.
    limit : int
        Max partners to return *per* queried gene.
    required_score : int
        STRING's minimum combined score, 0–1000 (400 = "medium confidence").
    fetch_post : callable, optional
        ``(url, fields) -> bytes`` override for tests.
    use_cache : bool
        Bypass the on-disk cache when False.

    Returns
    -------
    dict
        ``{species, genes, direct, partners}`` where *direct* is the combined
        score (0–1) of the queried pair if STRING links them (else ``None``) and
        *partners* is a list of :class:`Interaction` dicts, strongest first.
    """
    from . import biocache

    tax = _taxid(species)
    clean = []
    for g in genes:
        t = str(g).strip()
        if t and t not in clean:
            clean.append(t)
    if not clean:
        raise StringError("no gene identifiers supplied")

    cache_key = f"{tax}:{required_score}:{limit}:" + ",".join(sorted(clean))
    if use_cache:
        cached = biocache.get("interactions", cache_key)
        if cached is not None:
            return cached

    fetch_post = fetch_post or default_post_fetch()
    fields = {
        "identifiers": "\r".join(clean),  # STRING pairs ids with CR
        "species": str(tax),
        "limit": str(max(1, int(limit))),
        "required_score": str(max(0, min(1000, int(required_score)))),
        "caller_identity": _CALLER,
    }
    raw = fetch_post(f"{_STRING}/json/interaction_partners", fields)
    try:
        rows = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise StringError(f"invalid JSON from STRING: {exc}") from exc
    if not isinstance(rows, list):
        rows = []

    partners: List[Interaction] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        a = str(row.get("preferredName_A", "")).strip()
        b = str(row.get("preferredName_B", "")).strip()
        score = _num(row.get("score"))
        channels = {
            key: _num(row.get(key))
            for key in STRING_CHANNELS
            if row.get(key) is not None
        }
        partners.append(Interaction(query=a, partner=b, score=score, channels=channels))

    partners.sort(key=lambda it: it.score, reverse=True)

    # The direct pair score can't be read reliably off the top-N partner lists
    # (the edge may fall outside each gene's top N), so ask STRING's /network
    # endpoint — it returns only edges *among* the queried identifiers.
    direct: Optional[float] = None
    if len(clean) >= 2:
        direct = _direct_score(fetch_post, clean, tax)
    result = {
        "species": tax,
        "genes": clean,
        "direct": direct,
        "partners": [p.to_dict() for p in partners],
    }
    if use_cache:
        biocache.set("interactions", cache_key, result)
    return result
