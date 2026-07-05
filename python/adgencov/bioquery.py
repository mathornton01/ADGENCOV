"""External bio-database lookups — GEO term search + protein-ID translation.

Two small, self-contained clients that sit on the Python side of the language
boundary (like :mod:`adgencov.geo`): all network access lives here so the rest
of the stack — and the test suite — never touches the wire.

* :func:`search_geo` turns a free-text query ("asthma airway smooth muscle")
  into a ranked list of GEO **series** accessions (``GSE…``) via NCBI's
  E-utilities (``esearch`` + ``esummary`` on the ``gds`` database).  The result
  feeds straight into the GEO accession box of the dashboard / ``/analyze/geo``.
* :func:`translate_protein_ids` turns a batch of protein / gene identifiers into
  human-readable protein names via the UniProt REST API.  It auto-detects three
  common id spaces — UniProt accessions (``P04637``), numeric Entrez GeneIDs
  (``7157``), and gene symbols (``TP53``) — or you can pin one with *source*.

Both functions take an injectable ``fetch`` callable (``url -> bytes``) so tests
drive them from canned fixtures with **no** network round-trip.  The default
fetcher uses :mod:`urllib` with a short timeout and a descriptive User-Agent, as
NCBI and UniProt both request.

Example
-------
>>> from adgencov import bioquery
>>> hits = bioquery.search_geo("dexamethasone airway smooth muscle")   # doctest: +SKIP
>>> hits[0].accession                                                   # doctest: +SKIP
'GSE52778'
>>> names = bioquery.translate_protein_ids(["P04637", "7157", "TP53"])  # doctest: +SKIP
>>> names[0].name                                                       # doctest: +SKIP
'Cellular tumor antigen p53'
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "BioQueryError",
    "GeoSearchHit",
    "ProteinName",
    "search_geo",
    "translate_protein_ids",
    "default_fetch",
]

# A fetcher maps an absolute URL to raw response bytes.  Injectable for tests.
Fetch = Callable[[str], bytes]

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_UNIPROT = "https://rest.uniprot.org/uniprotkb/search"

# UniProt accession grammar (canonical 6- or 10-char form), e.g. P04637, A0A0B4J2F0.
_UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)
_DIGITS_RE = re.compile(r"^\d+$")
_USER_AGENT = "adgencov/1.0 (https://thorntonstatistical.com; mailto:support@thorntonstatistical.com)"


class BioQueryError(RuntimeError):
    """Raised when an external lookup cannot be performed or parsed."""


# ---------------------------------------------------------------------------
# Default network fetcher (the only place that touches the wire)
# ---------------------------------------------------------------------------
def default_fetch(timeout: float = 20.0) -> Fetch:
    """Return a urllib-based ``fetch(url) -> bytes`` with a fixed *timeout*."""

    def _fetch(url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - surface a clean domain error
            raise BioQueryError(f"request failed: {url}: {exc}") from exc

    return _fetch


def _fetch_json(fetch: Fetch, url: str) -> Any:
    raw = fetch(url)
    try:
        return json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise BioQueryError(f"invalid JSON from {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# GEO term search (NCBI E-utilities, gds database)
# ---------------------------------------------------------------------------
@dataclass
class GeoSearchHit:
    """One GEO **series** returned by :func:`search_geo`."""

    accession: str            # GSE…
    title: str
    summary: str = ""
    taxon: str = ""
    n_samples: int = 0
    gds_type: str = ""        # e.g. "Expression profiling by high throughput sequencing"
    platform: str = ""        # GPL id(s)
    pub_date: str = ""        # PDAT (yyyy/mm/dd)
    uid: str = ""             # NCBI internal UID
    url: str = ""             # public GEO landing page

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accession": self.accession,
            "title": self.title,
            "summary": self.summary,
            "taxon": self.taxon,
            "n_samples": self.n_samples,
            "gds_type": self.gds_type,
            "platform": self.platform,
            "pub_date": self.pub_date,
            "uid": self.uid,
            "url": self.url,
        }


def _geo_landing_url(accession: str) -> str:
    return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"


def search_geo(
    term: str,
    *,
    retmax: int = 20,
    series_only: bool = True,
    fetch: Optional[Fetch] = None,
    api_key: Optional[str] = None,
) -> List[GeoSearchHit]:
    """Search GEO for series matching a free-text *term*.

    Runs NCBI ``esearch`` (db=gds) to get UIDs, then ``esummary`` to hydrate
    each into a :class:`GeoSearchHit`.  When *series_only* (default) only
    ``GSE`` series are returned — the entry kind :func:`adgencov.geo` can
    actually analyze — with datasets/platforms/samples filtered out.

    Parameters
    ----------
    term : str
        Free-text query.  Standard Entrez syntax works, e.g.
        ``"asthma AND Homo sapiens[Organism]"``.
    retmax : int
        Maximum UIDs to request (1–100).
    series_only : bool
        Keep only ``entrytype == "GSE"`` hits.
    fetch : callable, optional
        ``url -> bytes`` override (tests inject a canned fetcher).
    api_key : str, optional
        NCBI API key to raise the rate limit (appended as ``api_key=``).
    """
    q = (term or "").strip()
    if not q:
        raise BioQueryError("empty search term")
    retmax = max(1, min(int(retmax), 100))
    fetch = fetch or default_fetch()

    common = f"&api_key={urllib.parse.quote(api_key)}" if api_key else ""
    es_url = (
        f"{_EUTILS}/esearch.fcgi?db=gds&retmode=json&retmax={retmax}"
        f"&term={urllib.parse.quote(q)}{common}"
    )
    es = _fetch_json(fetch, es_url)
    idlist = (es.get("esearchresult", {}) or {}).get("idlist", []) or []
    if not idlist:
        return []

    sum_url = (
        f"{_EUTILS}/esummary.fcgi?db=gds&retmode=json"
        f"&id={','.join(idlist)}{common}"
    )
    summ = _fetch_json(fetch, sum_url)
    result = summ.get("result", {}) or {}

    hits: List[GeoSearchHit] = []
    # Preserve esearch relevance order via the "uids" list esummary echoes back.
    order = result.get("uids", idlist) or idlist
    for uid in order:
        rec = result.get(uid)
        if not isinstance(rec, dict):
            continue
        entry = str(rec.get("entrytype", "")).upper()
        if series_only and entry != "GSE":
            continue
        accession = str(rec.get("accession", "")).strip()
        if not accession:
            # esummary sometimes omits the prefix; reconstruct from entrytype.
            accession = f"{entry}{uid[-6:].lstrip('0') or uid}"
        hits.append(
            GeoSearchHit(
                accession=accession,
                title=str(rec.get("title", "")).strip(),
                summary=str(rec.get("summary", "")).strip(),
                taxon=str(rec.get("taxon", "")).strip(),
                n_samples=_as_int(rec.get("n_samples")),
                gds_type=str(rec.get("gdstype", "")).strip(),
                platform=_format_gpl(rec.get("gpl")),
                pub_date=str(rec.get("pdat", "")).strip(),
                uid=str(uid),
                url=_geo_landing_url(accession),
            )
        )
    return hits


def _as_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _format_gpl(v: Any) -> str:
    if v in (None, ""):
        return ""
    parts = [p for p in re.split(r"[;,\s]+", str(v)) if p]
    return "; ".join(p if p.upper().startswith("GPL") else f"GPL{p}" for p in parts)


# ---------------------------------------------------------------------------
# Protein / gene id -> name translation (UniProt REST)
# ---------------------------------------------------------------------------
@dataclass
class ProteinName:
    """A resolved (or unresolved) identifier from :func:`translate_protein_ids`."""

    query: str                       # the id the caller asked about
    matched: bool = False
    name: str = ""                   # recommended protein full name
    gene: str = ""                   # primary gene symbol
    organism: str = ""               # scientific organism name
    accession: str = ""              # resolving UniProt accession
    source: str = ""                 # detected id space: uniprot | geneid | gene | ...
    url: str = ""                    # UniProt entry page

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "matched": self.matched,
            "name": self.name,
            "gene": self.gene,
            "organism": self.organism,
            "accession": self.accession,
            "source": self.source,
            "url": self.url,
        }


def classify_id(token: str) -> str:
    """Guess the id space of *token*: ``uniprot``, ``geneid``, or ``gene``."""
    t = token.strip()
    if _UNIPROT_RE.match(t.upper()):
        return "uniprot"
    if _DIGITS_RE.match(t):
        return "geneid"
    return "gene"


_UNIPROT_FIELDS = "accession,id,protein_name,gene_names,organism_name,xref_geneid"


def translate_protein_ids(
    ids: Sequence[str],
    *,
    source: str = "auto",
    reviewed_only: bool = True,
    fetch: Optional[Fetch] = None,
) -> List[ProteinName]:
    """Translate protein / gene identifiers to protein names via UniProt.

    Parameters
    ----------
    ids : sequence of str
        Identifiers to resolve.  Duplicates and blanks are ignored; order of the
        first occurrence is preserved in the output.
    source : {"auto", "uniprot", "geneid", "gene"}
        Id space.  ``auto`` (default) classifies each id individually — UniProt
        accession, numeric Entrez GeneID, or gene symbol.
    reviewed_only : bool
        Restrict GeneID / gene-symbol lookups to reviewed (Swiss-Prot) entries so
        a symbol resolves to its canonical protein rather than many isoforms.
    fetch : callable, optional
        ``url -> bytes`` override for tests.

    Returns
    -------
    list of ProteinName
        One entry per unique input id (``matched=False`` when unresolved).
    """
    fetch = fetch or default_fetch()
    source = (source or "auto").lower()
    if source not in ("auto", "uniprot", "geneid", "gene"):
        raise BioQueryError(f"unknown source {source!r}")

    # De-dup, keep first-seen order.
    seen: Dict[str, None] = {}
    for raw in ids:
        tok = str(raw).strip()
        if tok and tok not in seen:
            seen[tok] = None
    tokens = list(seen.keys())
    if not tokens:
        return []

    # Bucket ids by the id space we'll query them in.
    buckets: Dict[str, List[str]] = {"uniprot": [], "geneid": [], "gene": []}
    kind_of: Dict[str, str] = {}
    for tok in tokens:
        kind = classify_id(tok) if source == "auto" else source
        kind_of[tok] = kind
        buckets[kind].append(tok)

    resolved: Dict[str, ProteinName] = {}
    for kind, group in buckets.items():
        if not group:
            continue
        _resolve_bucket(fetch, kind, group, reviewed_only, resolved)

    # Emit in original order, filling unmatched placeholders.
    out: List[ProteinName] = []
    for tok in tokens:
        out.append(
            resolved.get(tok)
            or ProteinName(query=tok, matched=False, source=kind_of[tok])
        )
    return out


def _resolve_bucket(
    fetch: Fetch,
    kind: str,
    group: List[str],
    reviewed_only: bool,
    resolved: Dict[str, ProteinName],
) -> None:
    """Query UniProt for one id-space bucket, filling *resolved* in place."""
    query = _build_query(kind, group, reviewed_only)
    url = (
        f"{_UNIPROT}?query={urllib.parse.quote(query)}"
        f"&fields={_UNIPROT_FIELDS}&format=json&size=500"
    )
    data = _fetch_json(fetch, url)
    for rec in data.get("results", []) or []:
        _match_record(kind, group, rec, resolved)


def _build_query(kind: str, group: List[str], reviewed_only: bool) -> str:
    if kind == "uniprot":
        return " OR ".join(f"accession:{t.upper()}" for t in group)
    rev = " AND reviewed:true" if reviewed_only else ""
    if kind == "geneid":
        inner = " OR ".join(f"xref:geneid-{t}" for t in group)
        return f"({inner}){rev}"
    # gene symbols
    inner = " OR ".join(f"gene:{t}" for t in group)
    return f"({inner}){rev}"


def _match_record(
    kind: str,
    group: List[str],
    rec: Dict[str, Any],
    resolved: Dict[str, ProteinName],
) -> None:
    """Attach a UniProt record to whichever input id(s) it answers."""
    acc = str(rec.get("primaryAccession", "")).strip()
    name = (
        rec.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "")
    )
    if not name:  # fall back to submission name when no recommended name exists
        subs = rec.get("proteinDescription", {}).get("submissionNames", [])
        if subs:
            name = subs[0].get("fullName", {}).get("value", "")
    genes = rec.get("genes", []) or []
    gene = ""
    gene_syns = set()
    if genes:
        gene = genes[0].get("geneName", {}).get("value", "")
        for g in genes:
            gn = g.get("geneName", {}).get("value")
            if gn:
                gene_syns.add(gn.upper())
    organism = rec.get("organism", {}).get("scientificName", "")

    made = ProteinName(
        query="",
        matched=True,
        name=name,
        gene=gene,
        organism=organism,
        accession=acc,
        source=kind,
        url=f"https://www.uniprot.org/uniprotkb/{acc}/entry" if acc else "",
    )

    # Figure out which requested id(s) this record satisfies.
    keys: List[str] = []
    if kind == "uniprot":
        secondary = {s.upper() for s in rec.get("secondaryAccessions", []) or []}
        for t in group:
            tu = t.upper()
            if tu == acc.upper() or tu in secondary:
                keys.append(t)
    elif kind == "geneid":
        xref_ids = {
            str(x.get("id"))
            for x in rec.get("uniProtKBCrossReferences", []) or []
            if x.get("database") == "GeneID"
        }
        for t in group:
            if t in xref_ids:
                keys.append(t)
    else:  # gene symbols
        for t in group:
            if t.upper() in gene_syns:
                keys.append(t)

    for t in keys:
        # First reviewed/nonempty hit wins; don't overwrite a good match.
        if t not in resolved:
            resolved[t] = ProteinName(**{**made.__dict__, "query": t})
