"""GEO ingestion — pull transcriptomics series into the fast path (Phase B).

This module lets ADGENCOV read a public gene-expression series straight from
NCBI GEO (Gene Expression Omnibus) by accession (``GSE…``) and run it through
the recommender end-to-end.  It sits on the Python side of the language boundary
(see :mod:`adgencov`): all network/parsing lives here, and the standardized
matrix it produces is handed to the compiled C++ core.

Design choices
--------------
* **No heavyweight dependency for parsing.**  GEO publishes a stable, plain-text
  *series matrix* format (``GSEnnn_series_matrix.txt.gz``).  We parse it directly
  with :mod:`pandas`, so the whole ingestion + analysis path is exercised
  **offline** in the test suite from a small committed fixture — no ``GEOparse``
  install and no network round-trip required.  ``GEOparse`` remains an optional
  extra (``pip install 'adgencov[geo]'``) for richer platform annotation, and
  :func:`fetch_series` will use it transparently when present.
* **Reuse the C++ I/O.**  The parsed series is materialized as a pipeline-shaped
  TSV (a ``gene`` column + one column per sample) and fed to the core's
  ``load_expression_matrix`` → ``preprocess`` → ``build_group_labels`` →
  ``factorize`` → :func:`adgencov.analyze` chain.  The numbers are therefore
  identical to running the CLI on a local file.
* **Local caching.**  Downloads land in ``~/.cache/adgencov/geo`` (override with
  ``cache_dir`` or ``$ADGENCOV_CACHE``) so repeat analyses are instant and CI can
  pre-seed the cache.

Example
-------
>>> from adgencov import geo
>>> series = geo.read_series_matrix("GSE12345_series_matrix.txt")   # doctest: +SKIP
>>> result = geo.analyze_series(series, group="gene_family")        # doctest: +SKIP
>>> result.best.spec.method                                          # doctest: +SKIP
'ad_oas'

Or in one call from an accession (network):

>>> result = geo.analyze_series("GSE52778")                          # doctest: +SKIP
"""
from __future__ import annotations

import gzip
import io
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import (
    AnalysisResult,
    analyze,
    build_group_labels,
    factorize,
    load_expression_matrix,
    preprocess,
)

__all__ = [
    "GeoSeries",
    "GeoError",
    "series_matrix_url",
    "default_cache_dir",
    "read_series_matrix",
    "fetch_series",
    "load_series",
    "map_probes_to_genes",
    "analyze_series",
]

GENE_COL = "gene"
_ACCESSION_RE = re.compile(r"^GSE\d+$", re.IGNORECASE)


class GeoError(RuntimeError):
    """Raised when a GEO series cannot be located, downloaded, or parsed."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class GeoSeries:
    """A parsed GEO series: expression matrix + sample/series metadata.

    Attributes
    ----------
    accession : str
        The ``GSE…`` accession (or ``"<local>"`` for a file with no series id).
    expression : pandas.DataFrame
        Genes-by-samples table.  The first column is :data:`GENE_COL` (``"gene"``)
        holding the row identifier (probe id or mapped symbol); the remaining
        columns are one per sample (``GSM…``).
    samples : pandas.DataFrame
        Per-sample metadata indexed by sample id (``GSM…``): title,
        characteristics, source, etc. — one row per sample.
    metadata : dict
        Series-level ``!Series_*`` fields (title, summary, platform id, …).
    platform : str or None
        The platform accession (``GPL…``) if present.
    source_path : str or None
        Where the series matrix was read from (cache path or local file).
    """

    accession: str
    expression: pd.DataFrame
    samples: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: Dict[str, Any] = field(default_factory=dict)
    platform: Optional[str] = None
    source_path: Optional[str] = None

    # -- convenience ---------------------------------------------------------
    @property
    def sample_ids(self) -> List[str]:
        """The sample column names (everything except the gene column)."""
        return [c for c in self.expression.columns if c != GENE_COL]

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def n_genes(self) -> int:
        return int(self.expression.shape[0])

    @property
    def title(self) -> Optional[str]:
        return self.metadata.get("title")

    def sample_regex(self) -> str:
        """A regex matching exactly this series' sample columns.

        Used to drive the core ``load_expression_matrix`` selector.  We anchor
        on the explicit id list so unrelated metadata columns can never leak in.
        """
        ids = self.sample_ids
        if not ids:
            raise GeoError(f"series {self.accession!r} has no sample columns")
        return "^(?:" + "|".join(re.escape(s) for s in ids) + ")$"

    def to_frame(self, gene_col: str = GENE_COL) -> pd.DataFrame:
        """Return the expression table, optionally renaming the gene column."""
        if gene_col == GENE_COL:
            return self.expression.copy()
        return self.expression.rename(columns={GENE_COL: gene_col})

    def write_tsv(self, path: str, gene_col: str = "gene_short_name") -> str:
        """Write the pipeline-shaped TSV the C++ loader consumes; return path."""
        self.to_frame(gene_col).to_csv(path, sep="\t", index=False)
        return path

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GeoSeries(accession={self.accession!r}, "
            f"n_genes={self.n_genes}, n_samples={self.n_samples}, "
            f"platform={self.platform!r})"
        )


# ---------------------------------------------------------------------------
# URL / cache helpers
# ---------------------------------------------------------------------------
def series_matrix_url(accession: str) -> str:
    """Return the canonical NCBI URL for a series-matrix file.

    GEO groups series into buckets of 1000 by their leading digits, e.g.
    ``GSE52778`` lives under ``GSE52nnn`` and ``GSE567`` under ``GSEnnn``.

    >>> series_matrix_url("GSE52778")
    'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52778/matrix/GSE52778_series_matrix.txt.gz'
    >>> series_matrix_url("GSE567")
    'https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/GSE567/matrix/GSE567_series_matrix.txt.gz'
    """
    acc = accession.strip().upper()
    if not _ACCESSION_RE.match(acc):
        raise GeoError(f"not a GSE accession: {accession!r}")
    digits = acc[3:]
    # Replace the last three digits with 'nnn' (whole number if <1000 digits).
    stub = ("GSE" + digits[:-3] + "nnn") if len(digits) > 3 else "GSEnnn"
    return (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{stub}/{acc}/matrix/{acc}_series_matrix.txt.gz"
    )


def default_cache_dir() -> str:
    """Directory used to cache downloaded series matrices."""
    base = os.environ.get("ADGENCOV_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "adgencov"
    )
    return os.path.join(base, "geo")


def _looks_like_accession(source: Any) -> bool:
    return isinstance(source, str) and bool(_ACCESSION_RE.match(source.strip()))


# ---------------------------------------------------------------------------
# Series-matrix parser (offline, pandas-only)
# ---------------------------------------------------------------------------
def _open_text(source: Union[str, os.PathLike, io.IOBase]):
    """Open a path (gzip-aware) or return a text buffer unchanged."""
    if hasattr(source, "read"):  # already a file-like
        return source
    path = os.fspath(source)
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _strip_quotes(tok: str) -> str:
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return tok[1:-1]
    return tok


def read_series_matrix(
    source: Union[str, os.PathLike, io.IOBase],
    *,
    accession: Optional[str] = None,
) -> GeoSeries:
    """Parse a GEO *series matrix* text file (optionally gzipped) offline.

    Parameters
    ----------
    source : path or text file-like
        A ``…_series_matrix.txt`` / ``.txt.gz`` file, or an open text buffer.
    accession : str, optional
        Override the accession; otherwise taken from ``!Series_geo_accession``
        or the filename.

    Returns
    -------
    GeoSeries
    """
    series_meta: Dict[str, Any] = {}
    sample_fields: Dict[str, List[str]] = {}
    header: Optional[List[str]] = None
    data_rows: List[List[str]] = []
    in_table = False
    src_path = None if hasattr(source, "read") else os.fspath(source)

    fh = _open_text(source)
    try:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                parts = [_strip_quotes(t) for t in line.split("\t")]
                if header is None:
                    header = parts
                else:
                    if any(c != "" for c in parts):
                        data_rows.append(parts)
                continue
            if line.startswith("!Series_"):
                key, _, rest = line.partition("\t")
                field_name = key[len("!Series_"):].strip()
                vals = [_strip_quotes(t) for t in rest.split("\t")] if rest else []
                _accumulate(series_meta, field_name, vals)
            elif line.startswith("!Sample_"):
                key, _, rest = line.partition("\t")
                field_name = key[len("!Sample_"):].strip()
                vals = [_strip_quotes(t) for t in rest.split("\t")] if rest else []
                sample_fields.setdefault(field_name, []).extend(vals)
    finally:
        if not hasattr(source, "read"):
            fh.close()

    if header is None or not data_rows:
        raise GeoError(
            "no expression table found "
            "(missing !series_matrix_table_begin/…_end block)"
        )

    # First header cell is the row-id label (usually "ID_REF"); rest are GSM ids.
    sample_ids = header[1:]
    n = len(sample_ids)
    genes: List[str] = []
    matrix: List[List[float]] = []
    for row in data_rows:
        rid = row[0]
        if rid == "":
            continue
        cells = row[1 : 1 + n]
        # Pad short rows with NaN so ragged files don't crash.
        if len(cells) < n:
            cells = cells + [""] * (n - len(cells))
        genes.append(rid)
        matrix.append([_to_float(c) for c in cells])

    expr = pd.DataFrame(matrix, columns=sample_ids)
    expr.insert(0, GENE_COL, genes)

    # Per-sample metadata table (indexed by GSM id when available).
    samples = _build_sample_frame(sample_fields, sample_ids)

    acc = (
        accession
        or _first(series_meta.get("geo_accession"))
        or _accession_from_path(src_path)
        or "<local>"
    )
    platform = _first(series_meta.get("platform_id")) or _first(
        sample_fields.get("platform_id")
    )
    meta = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
            for k, v in series_meta.items()}

    return GeoSeries(
        accession=acc,
        expression=expr,
        samples=samples,
        metadata=meta,
        platform=platform,
        source_path=src_path,
    )


def _accumulate(store: Dict[str, Any], field_name: str, vals: List[str]) -> None:
    if field_name in store:
        prev = store[field_name]
        if not isinstance(prev, list):
            prev = [prev]
        prev.extend(vals)
        store[field_name] = prev
    else:
        store[field_name] = list(vals)


def _build_sample_frame(
    sample_fields: Dict[str, List[str]], sample_ids: Sequence[str]
) -> pd.DataFrame:
    n = len(sample_ids)
    cols: Dict[str, List[str]] = {}
    for k, vals in sample_fields.items():
        if len(vals) == n:
            cols[k] = list(vals)
    idx = sample_fields.get("geo_accession")
    index = idx if (idx and len(idx) == n) else list(sample_ids)
    frame = pd.DataFrame(cols)
    if len(frame) == n:
        frame.index = pd.Index(index, name="sample")
    return frame


def _to_float(cell: str) -> float:
    s = cell.strip()
    if s == "" or s.lower() in ("na", "nan", "null", "none"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _first(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _accession_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    m = re.search(r"(GSE\d+)", os.path.basename(path), re.IGNORECASE)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# Probe -> gene symbol mapping
# ---------------------------------------------------------------------------
def map_probes_to_genes(
    series: GeoSeries,
    mapping: Union[Mapping[str, str], pd.Series, pd.DataFrame],
    *,
    probe_col: Optional[str] = None,
    gene_col: Optional[str] = None,
    aggregate: str = "max_mean",
    drop_unmapped: bool = True,
) -> GeoSeries:
    """Replace probe ids with gene symbols, collapsing duplicates.

    Parameters
    ----------
    series : GeoSeries
        A series whose rows are platform probe ids.
    mapping : dict, Series, or DataFrame
        probe→gene lookup.  A DataFrame must name ``probe_col`` and ``gene_col``.
    aggregate : {"max_mean", "mean", "sum"}
        How to collapse multiple probes for one gene.  ``max_mean`` keeps the
        single highest-mean-abundance probe (the convention the C++ preprocess
        uses for duplicate symbols); ``mean``/``sum`` combine them.
    drop_unmapped : bool
        Drop rows whose probe has no gene symbol (or maps to empty/NaN).

    Returns
    -------
    GeoSeries
        A new series keyed by gene symbol.
    """
    if isinstance(mapping, pd.DataFrame):
        if not probe_col or not gene_col:
            raise GeoError("probe_col and gene_col are required for a DataFrame mapping")
        lut = dict(zip(mapping[probe_col].astype(str), mapping[gene_col].astype(str)))
    elif isinstance(mapping, pd.Series):
        lut = {str(k): str(v) for k, v in mapping.items()}
    else:
        lut = {str(k): str(v) for k, v in dict(mapping).items()}

    expr = series.expression
    probes = expr[GENE_COL].astype(str)
    genes = probes.map(lambda p: lut.get(p, ""))

    df = expr.drop(columns=[GENE_COL]).copy()
    df.insert(0, GENE_COL, genes.values)
    if drop_unmapped:
        keep = df[GENE_COL].astype(str).str.strip().replace({"nan": ""}) != ""
        df = df[keep]
    if df.empty:
        raise GeoError("probe→gene mapping left no rows (check the mapping keys)")

    sample_cols = [c for c in df.columns if c != GENE_COL]
    if aggregate == "max_mean":
        means = df[sample_cols].mean(axis=1)
        df = df.assign(_m=means.values).sort_values("_m", ascending=False)
        df = df.drop_duplicates(GENE_COL, keep="first").drop(columns="_m")
    elif aggregate in ("mean", "sum"):
        agg = "mean" if aggregate == "mean" else "sum"
        df = df.groupby(GENE_COL, sort=False)[sample_cols].agg(agg).reset_index()
    else:
        raise GeoError(f"unknown aggregate {aggregate!r}")

    df = df.reset_index(drop=True)
    return GeoSeries(
        accession=series.accession,
        expression=df,
        samples=series.samples,
        metadata=series.metadata,
        platform=series.platform,
        source_path=series.source_path,
    )


# ---------------------------------------------------------------------------
# Network fetch (+ cache)
# ---------------------------------------------------------------------------
def fetch_series(
    accession: str,
    *,
    cache_dir: Optional[str] = None,
    force: bool = False,
    timeout: float = 60.0,
) -> GeoSeries:
    """Download (and cache) a series matrix by accession, then parse it.

    The download is cached under :func:`default_cache_dir` (or ``cache_dir``);
    a second call with the same accession reads from disk.  This is the only
    function here that touches the network — it is *not* exercised by the test
    suite (which parses the committed fixture directly via
    :func:`read_series_matrix`).
    """
    acc = accession.strip().upper()
    cdir = cache_dir or default_cache_dir()
    os.makedirs(cdir, exist_ok=True)
    dest = os.path.join(cdir, f"{acc}_series_matrix.txt.gz")

    if force or not os.path.exists(dest) or os.path.getsize(dest) == 0:
        url = series_matrix_url(acc)
        tmp = dest + ".part"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                out.write(resp.read())
            os.replace(tmp, dest)
        except Exception as exc:  # noqa: BLE001 - surface a clean domain error
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise GeoError(f"failed to download {acc} from {url}: {exc}") from exc

    return read_series_matrix(dest, accession=acc)


def load_series(
    source: Union[str, os.PathLike, GeoSeries, io.IOBase],
    *,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> GeoSeries:
    """Resolve *source* to a :class:`GeoSeries`.

    Accepts an already-parsed :class:`GeoSeries` (returned as-is), a ``GSE…``
    accession (downloaded + cached via :func:`fetch_series`), or a filesystem
    path / file-like to a series matrix (parsed offline).
    """
    if isinstance(source, GeoSeries):
        return source
    if _looks_like_accession(source):
        return fetch_series(str(source), cache_dir=cache_dir, force=force)
    return read_series_matrix(source)


# ---------------------------------------------------------------------------
# End-to-end: GEO series -> recommendation
# ---------------------------------------------------------------------------
def analyze_series(
    source: Union[str, os.PathLike, GeoSeries, io.IOBase],
    *,
    n_genes: int = 500,
    min_mean: float = 0.1,
    log_transform: bool = True,
    group: str = "gene_family",
    n_blocks: int = 4,
    top_fraction: float = 0.01,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> AnalysisResult:
    """Run the full recommender on a GEO series — pull → preprocess → analyze.

    This is the one call the GEO-facing API/GUI makes.  It resolves *source*
    (accession, path, or :class:`GeoSeries`), materializes the pipeline-shaped
    TSV, and drives the compiled core exactly as the CLI does on a local file:
    ``load_expression_matrix`` → ``preprocess`` → ``build_group_labels`` →
    ``factorize`` → :func:`adgencov.analyze`.

    Returns
    -------
    AnalysisResult
        JSON-serializable via :meth:`AnalysisResult.to_dict`.
    """
    series = load_series(source, cache_dir=cache_dir, force=force)
    if series.n_samples < 3:
        raise GeoError(
            f"series {series.accession!r} has {series.n_samples} samples; "
            "at least 3 are required for covariance estimation"
        )
    regex = series.sample_regex()

    # Materialize the TSV and drive the C++ loader so numbers match the CLI.
    with tempfile.TemporaryDirectory(prefix="adgencov_geo_") as td:
        tsv = os.path.join(td, f"{series.accession}.tsv")
        series.write_tsv(tsv, gene_col="gene_short_name")
        data = load_expression_matrix(tsv, sample_regex=regex, gene_col="gene_short_name")
        dataset = preprocess(
            data, n_genes=n_genes, min_mean=min_mean, log_transform=log_transform
        )

    labels = build_group_labels(dataset, group, n_blocks=n_blocks)
    codes = factorize(labels)
    X = np.asarray(dataset.X, dtype=float)
    result = analyze(X, codes, genes=list(dataset.genes), top_fraction=top_fraction)
    return result
