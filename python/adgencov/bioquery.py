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
    "GeneSymbol",
    "search_geo",
    "translate_protein_ids",
    "translate_gene_symbols",
    "rna_type_of",
    "default_fetch",
    "default_post_fetch",
]

# A fetcher maps an absolute URL to raw response bytes.  Injectable for tests.
Fetch = Callable[[str], bytes]
# A POST fetcher maps (url, form-fields) to raw response bytes.  Injectable too.
PostFetch = Callable[[str, Dict[str, str]], bytes]

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
_MYGENE = "https://mygene.info/v3/query"

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


def default_post_fetch(timeout: float = 20.0) -> PostFetch:
    """Return a urllib-based ``fetch(url, fields) -> bytes`` that POSTs a form.

    Used for batch APIs (mygene.info, STRING-db) that accept many identifiers in
    a single ``application/x-www-form-urlencoded`` body instead of the URL.
    """

    def _fetch(url: str, fields: Dict[str, str]) -> bytes:
        body = urllib.parse.urlencode(fields).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )
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


# ---------------------------------------------------------------------------
# Gene id -> HGNC symbol translation (mygene.info batch)
# ---------------------------------------------------------------------------
# mygene is faster and cleaner than UniProt for *pure symbol* mapping: one POST
# resolves a whole batch of Entrez / Ensembl / probe / alias ids to the current
# official symbol, and it returns ``type_of_gene`` so we can shape nodes by RNA
# class (protein-coding vs miRNA vs snoRNA) in the network view.
_MYGENE_FIELDS = "symbol,name,taxid,type_of_gene,entrezgene,ensembl.gene"
# Scopes are chosen *per id type* — searching a bare number across symbol/name
# scopes lets a spurious text match (e.g. "7157" → MIR7157) outrank the true
# Entrez hit, so we bucket ids and pin each bucket to the right id space.
_SCOPE_ENTREZ = "entrezgene,retired"
_SCOPE_ENSEMBL = "ensembl.gene,ensembl.transcript,ensembl.protein"
# ``wormbase`` lets C. elegans WormBase ids (WBGene…) and systematic sequence
# names (e.g. F35G12.3) resolve; it is inert for other species' ids.  The
# free-text ``name``/``other_names`` scopes are deliberately excluded: a query
# like "daf-16" would otherwise match a *different* gene whose description
# merely contains "DAF-16" (e.g. dod-3, "Downstream Of DAF-16").
_SCOPE_SYMBOL = "symbol,alias,uniprot,accession,refseq,wormbase,retired"
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*[GTP]\d{6,}", re.IGNORECASE)


def _mygene_scope_for(token: str) -> str:
    """Pick the mygene scope list appropriate to an id's shape."""
    t = token.strip()
    if _DIGITS_RE.match(t):
        return _SCOPE_ENTREZ
    if _ENSEMBL_RE.match(t):
        return _SCOPE_ENSEMBL
    return _SCOPE_SYMBOL
# NCBI/Ensembl species tokens accepted by mygene's ``species`` param.  C.
# elegans is passed as its taxid (6239) so mygene disambiguates worm genes;
# combined with the ``wormbase`` scope below this resolves WormBase ids
# (WBGene…), systematic sequence names (F35G12.3), and CGC symbols (daf-16).
_SPECIES_ALIASES = {
    "9606": "human", "human": "human", "homo sapiens": "human",
    "10090": "mouse", "mouse": "mouse", "mus musculus": "mouse",
    "10116": "rat", "rat": "rat", "rattus norvegicus": "rat",
    "6239": "6239", "worm": "6239", "celegans": "6239", "c. elegans": "6239",
    "c elegans": "6239", "caenorhabditis elegans": "6239", "nematode": "6239",
}


@dataclass
class GeneSymbol:
    """A resolved (or unresolved) gene identifier from :func:`translate_gene_symbols`."""

    query: str                       # the id the caller asked about
    matched: bool = False
    symbol: str = ""                 # current official (HGNC) symbol
    name: str = ""                   # full gene name
    rna_type: str = ""               # protein_coding | miRNA | snoRNA | ncRNA | …
    type_of_gene: str = ""           # raw mygene biotype (e.g. "protein-coding")
    entrez: str = ""                 # Entrez GeneID
    ensembl: str = ""                # Ensembl gene id
    taxid: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "matched": self.matched,
            "symbol": self.symbol,
            "name": self.name,
            "rna_type": self.rna_type,
            "type_of_gene": self.type_of_gene,
            "entrez": self.entrez,
            "ensembl": self.ensembl,
            "taxid": self.taxid,
        }


# Symbol prefixes that pin an RNA class regardless of (or absent) biotype.
_MIRNA_RE = re.compile(r"^(HSA-)?(MIR|LET-?7|MIRLET)", re.IGNORECASE)
_SNORNA_RE = re.compile(r"^(SNORD|SNORA|SCARNA|SNAR)", re.IGNORECASE)


def rna_type_of(symbol: str, type_of_gene: str = "") -> str:
    """Classify a gene into an RNA class for node shaping.

    Prefers mygene's authoritative ``type_of_gene`` biotype; falls back to a
    symbol-prefix heuristic (``MIR*`` → miRNA, ``SNORD*``/``SNORA*`` → snoRNA)
    when the biotype is missing or generic.  Returns one of ``protein_coding``,
    ``miRNA``, ``snoRNA``, ``ncRNA``, ``pseudogene``, or ``other``.
    """
    tog = (type_of_gene or "").strip().lower().replace("-", "_")
    sym = (symbol or "").strip()
    if tog in ("mirna", "micro_rna"):
        return "miRNA"
    if tog in ("snorna",):
        return "snoRNA"
    if _MIRNA_RE.match(sym):
        return "miRNA"
    if _SNORNA_RE.match(sym):
        return "snoRNA"
    if tog == "protein_coding":
        return "protein_coding"
    if tog in ("ncrna", "scrna", "snrna", "rrna", "trna", "lincrna", "lncrna"):
        return "ncRNA"
    if tog and "pseudo" in tog:
        return "pseudogene"
    return "other" if not tog else tog


def _norm_species(species: str) -> str:
    s = (species or "human").strip().lower()
    return _SPECIES_ALIASES.get(s, s)


def translate_gene_symbols(
    ids: Sequence[str],
    *,
    species: str = "human",
    scopes: Optional[str] = None,
    fetch_post: Optional[PostFetch] = None,
    use_cache: bool = True,
) -> List[GeneSymbol]:
    """Batch-resolve gene identifiers to their current official symbols.

    Sends one mygene.info ``/query`` POST for the whole (de-duplicated) batch —
    Entrez GeneIDs, Ensembl ids, probe ids, aliases, or already-official symbols
    all resolve to the canonical HGNC symbol plus a coarse RNA class.  Results
    are memoized per ``(species, id)`` in the on-disk cache so repeat lookups —
    e.g. re-rendering the same GEO series — never touch the wire.

    Parameters
    ----------
    ids : sequence of str
        Identifiers to resolve; duplicates and blanks are ignored, first-seen
        order is preserved in the output.
    species : str
        ``human`` (default), ``mouse``, ``rat``, a taxid, or any token mygene
        accepts.  Confines matches to that organism.
    scopes : str, optional
        Override the id spaces searched (comma-separated mygene scopes).
    fetch_post : callable, optional
        ``(url, fields) -> bytes`` override for tests.
    use_cache : bool
        When False, bypasses the on-disk cache (forces a fresh lookup).

    Returns
    -------
    list of GeneSymbol
        One entry per unique input id (``matched=False`` when unresolved).
    """
    from . import biocache

    sp = _norm_species(species)

    # De-dup, keep first-seen order.
    seen: Dict[str, None] = {}
    for raw in ids:
        tok = str(raw).strip()
        if tok and tok not in seen:
            seen[tok] = None
    tokens = list(seen.keys())
    if not tokens:
        return []

    ns = "symbols"
    resolved: Dict[str, GeneSymbol] = {}
    to_query = tokens
    if use_cache:
        hits, misses = biocache.get_many(ns, [f"{sp}:{t}" for t in tokens])
        for key, val in hits.items():
            tok = key.split(":", 1)[1]
            resolved[tok] = GeneSymbol(**val)
        to_query = [k.split(":", 1)[1] for k in misses]

    if to_query:
        fetch_post = fetch_post or default_post_fetch()
        # Bucket ids by scope so each id space is queried unambiguously.  An
        # explicit *scopes* override collapses everything into one bucket.
        buckets: Dict[str, List[str]] = {}
        for tok in to_query:
            scope = scopes if scopes else _mygene_scope_for(tok)
            buckets.setdefault(scope, []).append(tok)

        fresh: Dict[str, Dict[str, Any]] = {}
        for scope, group in buckets.items():
            fields = {
                "q": ",".join(group),
                "scopes": scope,
                "fields": _MYGENE_FIELDS,
                "species": sp,
            }
            raw = fetch_post(_MYGENE, fields)
            try:
                records = json.loads(raw)
            except (ValueError, TypeError) as exc:
                raise BioQueryError(f"invalid JSON from mygene: {exc}") from exc
            if not isinstance(records, list):
                records = [records] if isinstance(records, dict) else []
            _parse_mygene_records(records, sp, resolved, fresh)

        if use_cache and fresh:
            biocache.set_many(ns, fresh)

    # Emit in original order, filling unmatched placeholders.
    out: List[GeneSymbol] = []
    for tok in tokens:
        out.append(resolved.get(tok) or GeneSymbol(query=tok, matched=False))
    return out


def _parse_mygene_records(
    records: List[Any],
    species: str,
    resolved: Dict[str, "GeneSymbol"],
    fresh: Dict[str, Dict[str, Any]],
) -> None:
    """Fold a mygene response into *resolved* / *fresh* (first hit per query wins)."""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        q = str(rec.get("query", "")).strip()
        if not q or rec.get("notfound"):
            continue
        sym = str(rec.get("symbol", "")).strip()
        # mygene returns several hits per query ordered best-first. Keep the
        # first (highest _score) hit, but let a later *exact* symbol match
        # (symbol == the queried token) override an earlier fuzzy one, so e.g.
        # querying "daf-16" resolves to daf-16 rather than a same-batch alias hit.
        exact = bool(sym) and sym.lower() == q.lower()
        existing = resolved.get(q)
        if existing is not None:
            existing_exact = bool(existing.symbol) and existing.symbol.lower() == q.lower()
            if existing_exact or not exact:
                continue
        tog = str(rec.get("type_of_gene", "")).strip()
        ens = rec.get("ensembl")
        ens_id = ""
        if isinstance(ens, dict):
            ens_id = str(ens.get("gene", ""))
        elif isinstance(ens, list) and ens:
            ens_id = str((ens[0] or {}).get("gene", ""))
        gs = GeneSymbol(
            query=q,
            matched=bool(sym),
            symbol=sym,
            name=str(rec.get("name", "")),
            rna_type=rna_type_of(sym, tog),
            type_of_gene=tog,
            entrez=str(rec.get("entrezgene", "") or ""),
            ensembl=ens_id,
            taxid=_as_int(rec.get("taxid")),
        )
        resolved[q] = gs
        fresh[f"{species}:{q}"] = gs.to_dict()
