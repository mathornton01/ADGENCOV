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
    "supplementary_dir_url",
    "list_supplementary_files",
    "pick_supplementary_matrix",
    "rank_supplementary_matrices",
    "read_supplementary_matrix",
    "read_tar_matrix",
    "fetch_supplementary_series",
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
    allow_supplementary: bool = True,
) -> GeoSeries:
    """Download (and cache) a series matrix by accession, then parse it.

    The download is cached under :func:`default_cache_dir` (or ``cache_dir``);
    a second call with the same accession reads from disk.  This is the only
    function here that touches the network — it is *not* exercised by the test
    suite (which parses the committed fixture directly via
    :func:`read_series_matrix`).

    Series-matrix vs. supplementary data
    ------------------------------------
    Microarray series (e.g. ``GSE2034``) publish their full expression table
    *inside* the series matrix between the
    ``!series_matrix_table_begin/…_end`` markers.  **RNA-seq** series almost
    never do — their series matrix ships an *empty* table and the actual
    counts/FPKM/TPM values live in a supplementary file
    (``…_FPKM_Matrix.txt.gz``, ``…_raw_counts.tsv.gz``, …).  When the series
    matrix has no data rows and ``allow_supplementary`` is true, we
    transparently fall back to :func:`fetch_supplementary_series`, which finds
    and parses the best matrix-shaped supplementary file.  Set
    ``allow_supplementary=False`` to require an in-matrix table.
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

    try:
        return read_series_matrix(dest, accession=acc)
    except GeoError as exc:
        if not allow_supplementary:
            raise
        # Empty in-matrix table (typical for RNA-seq): fall back to the
        # supplementary FPKM/TPM/counts matrix, carrying series metadata over.
        try:
            meta = _series_metadata_only(dest)
            return fetch_supplementary_series(
                acc,
                cache_dir=cdir,
                force=force,
                timeout=timeout,
                series_metadata=meta,
            )
        except GeoError as exc2:
            raise GeoError(
                f"{acc}: series matrix carries no expression table and no usable "
                f"supplementary matrix was found ({exc2})"
            ) from exc


# ---------------------------------------------------------------------------
# Supplementary-file fallback (RNA-seq counts / FPKM / TPM matrices)
# ---------------------------------------------------------------------------
# Columns we never treat as samples: structural annotation emitted by the
# common RNA-seq quantifiers (Cufflinks/cuffdiff, featureCounts, salmon, …).
_ANNOT_NAME_DENY = frozenset(
    {
        "gene_id", "geneid", "gene", "gene_short_name", "gene_symbol",
        "gene_name", "genename", "symbol", "tracking_id", "transcript_id",
        "transcript", "tx_id", "ensembl", "ensembl_gene_id", "entrez",
        "entrezid", "refseq", "id", "id_ref", "name", "probe", "probe_id",
        "class_code", "nearest_ref_id", "tss_id", "locus", "length",
        "coverage", "chr", "chrom", "chromosome", "start", "end", "strand",
        "width", "biotype", "gene_biotype", "description", "geneid_version",
        # Differential-expression statistic columns emitted un-suffixed by
        # DESeq2/edgeR/limma-style annotated tables (matched by exact name so a
        # real sample can't be caught): fold-changes, p-values, flags, etc.
        "fc", "logfc", "log2fc", "log2foldchange", "foldchange", "fold_change",
        "pvalue", "p_value", "pval", "padj", "padjust", "padj_value",
        "qvalue", "qval", "fdr", "significant", "regulate", "regulation",
        "basemean", "base_mean", "lfcse", "stat", "tstat", "t_stat",
        "zscore", "z_score", "dispersion", "direction", "updown", "change",
    }
)
# Column-name suffixes that mark per-sample *statistics*, not expression.
_ANNOT_SUFFIX_DENY = (
    "_conf_lo", "_conf_hi", "_status", "_stat", "_pval", "_p_value",
    "_pvalue", "_qval", "_q_value", "_padj", "_fdr", "_log2fc",
    "_log2foldchange", "_foldchange", "_fold_change", "_test_stat",
    "_stderr", "_se", "_ci_lo", "_ci_hi", "_lo", "_hi",
)
# Per-sample expression-unit suffixes, in preference order (raw counts first,
# then normalized units).  A column ending in one of these is a real sample
# measurement; used both to isolate sample columns in annotated tables and to
# strip the unit tag from the derived sample name.
_EXPRESSION_SUFFIXES = (
    "_count", "_counts", "_fpkm", "_rpkm", "_cpm", "_tpm",
)
# Preferred gene-identifier column names, in priority order.
_GENE_COL_PRIORITY = (
    "gene_short_name", "gene_symbol", "gene_name", "symbol", "gene",
    "gene_id", "geneid", "ensembl_gene_id", "tracking_id", "transcript_id",
    "id_ref", "id", "name", "probe_id", "probe",
)
# File-name keywords that flag a matrix-shaped supplementary file, scored high.
_MATRIX_KEYWORDS = (
    ("fpkm", 6), ("tpm", 6), ("rpkm", 6), ("cpm", 5), ("counts", 5),
    ("count", 4), ("matrix", 4), ("expression", 4), ("abundance", 3),
    ("normalized", 2), ("norm", 1), ("genes", 1),
)
# File-name keywords that make a file *structurally* impossible as an expression
# matrix — sequence/annotation/prose formats.  These hard-disqualify a name.
_MATRIX_HARD_DENY = (
    "readme", "filelist", "file_list", "metadata", "sample_info",
    "design", "gtf", "gff", "bed", "fasta", "fastq",
    "supplementary_methods", "annotation.gtf", "annotation.gff",
)
# File-name keywords that *suggest* a differential-expression / annotated table
# rather than a bare matrix — BUT such files (DESeq2 results, cuffdiff, annotated
# quantification) very often still embed a per-sample count/TPM/FPKM matrix
# alongside the statistics.  So these only *penalize* the name; they no longer
# disqualify it.  Content-based validation (see ``fetch_supplementary_series``)
# is the final arbiter — we try to actually parse a matrix out of the file.
_MATRIX_SOFT_DENY = (
    "diff", "de_results", "deseq", "deg", "results", "annot", "meta",
)
_TABULAR_EXTS = (
    ".txt.gz", ".tsv.gz", ".csv.gz", ".tab.gz", ".txt", ".tsv", ".csv", ".tab",
)
# Archive extensions that may *bundle* an expression matrix (or per-sample
# files that assemble into one) — common for single-cell / array ``_RAW.tar``.
_TAR_EXTS = (".tar.gz", ".tgz", ".tar")


def _suppl_max_bytes() -> int:
    """Per-file supplementary download ceiling (MB), overridable via env.

    Multi-GB single-cell matrices (e.g. a 3 GB ``log2TPM`` file) would otherwise
    stream forever and never parse; we skip anything over the cap and move to a
    smaller candidate.  ``ADGENCOV_SUPPL_MAX_MB`` raises/lowers the limit.
    """
    raw = os.environ.get("ADGENCOV_SUPPL_MAX_MB")
    default = 700
    try:
        mb = float(raw) if raw else default
    except ValueError:
        mb = default
    return int(max(1.0, mb) * 1024 * 1024)


def _rm_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _download_capped(
    url: str, dest: str, *, timeout: float, max_bytes: int
) -> None:
    """Stream *url* to *dest*, aborting (and raising) if it exceeds *max_bytes*.

    Streaming (vs. ``resp.read()`` into RAM) keeps big matrices off the heap,
    and the byte ceiling turns an unbounded multi-GB download into a fast,
    clean skip.  Honors ``Content-Length`` up front when the server sends it.
    """
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            clen = resp.headers.get("Content-Length")
            if clen is not None:
                try:
                    if int(clen) > max_bytes:
                        raise GeoError(
                            f"file is {int(clen) // (1024 * 1024)} MB, over the "
                            f"{max_bytes // (1024 * 1024)} MB cap (raise "
                            "ADGENCOV_SUPPL_MAX_MB to allow)"
                        )
                except ValueError:
                    pass
            total = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise GeoError(
                            f"exceeded the {max_bytes // (1024 * 1024)} MB cap "
                            "mid-download (raise ADGENCOV_SUPPL_MAX_MB to allow)"
                        )
                    out.write(chunk)
        os.replace(tmp, dest)
    except GeoError:
        _rm_quiet(tmp)
        raise
    except Exception as exc:  # noqa: BLE001
        _rm_quiet(tmp)
        raise GeoError(f"download failed ({exc})") from exc


def supplementary_dir_url(accession: str) -> str:
    """Return the NCBI ``suppl/`` directory URL for a series accession."""
    acc = accession.strip().upper()
    if not _ACCESSION_RE.match(acc):
        raise GeoError(f"not a GSE accession: {accession!r}")
    digits = acc[3:]
    stub = ("GSE" + digits[:-3] + "nnn") if len(digits) > 3 else "GSEnnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{acc}/suppl/"


def list_supplementary_files(accession: str, *, timeout: float = 60.0) -> List[str]:
    """List supplementary file names published under a series' ``suppl/`` dir."""
    url = supplementary_dir_url(accession)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise GeoError(f"failed to list supplementary files at {url}: {exc}") from exc
    names: List[str] = []
    for m in re.finditer(r'href="([^"]+)"', html):
        href = m.group(1)
        # Skip parent-dir links, absolute URLs, and query links.
        if href.startswith(("/", "?", "http://", "https://")) or href in ("../",):
            continue
        name = href.rstrip("/")
        if name and name not in names:
            names.append(name)
    return names


def _score_matrix_candidate(name: str) -> int:
    """Score a supplementary file name as a probable expression matrix.

    Higher is better.  A score of ``-100`` means "structurally impossible"
    (non-tabular extension or a hard-deny format) and is never worth opening;
    any other score is a *plausible* candidate that content validation can
    confirm or reject.  Differential-expression / annotated tables get a soft
    penalty but remain candidates, because they routinely embed a per-sample
    count/TPM/FPKM matrix next to their statistics (e.g. a DESeq2 ``.annot``
    file with ``*_count`` columns).
    """
    low = name.lower()
    if not low.endswith(_TABULAR_EXTS):
        return -100
    if any(bad in low for bad in _MATRIX_HARD_DENY):
        return -100
    score = 0
    for kw, weight in _MATRIX_KEYWORDS:
        if kw in low:
            score += weight
    for bad in _MATRIX_SOFT_DENY:
        if bad in low:
            score -= 3
    # A bare "<acc>_something.txt.gz" with no keyword is still a plausible
    # matrix — give it a small floor so it beats clearly-annotation files.
    return score if score > 0 else 1


def rank_supplementary_matrices(names: Sequence[str]) -> List[str]:
    """Return all plausibly-matrix supplementary names, best candidate first.

    Anything not structurally disqualified (see :func:`_score_matrix_candidate`)
    is included, so a series whose only file is a differential-expression /
    annotated table is still offered up for content validation rather than
    rejected on its name alone.
    """
    scored = [(n, _score_matrix_candidate(n)) for n in names]
    plausible = [(n, s) for n, s in scored if s > -100]
    plausible.sort(key=lambda ns: (ns[1], -len(ns[0])), reverse=True)
    return [n for n, _ in plausible]


def pick_supplementary_matrix(names: Sequence[str]) -> Optional[str]:
    """Choose the most matrix-like supplementary file name, or ``None``."""
    ranked = rank_supplementary_matrices(names)
    return ranked[0] if ranked else None


def _sniff_delimiter(sample_line: str) -> str:
    """Guess the field delimiter of a header line (tab, comma, or whitespace)."""
    if "\t" in sample_line:
        return "\t"
    if "," in sample_line and sample_line.count(",") >= 2:
        return ","
    return r"\s+"


def _mostly_numeric(values: Sequence[str], *, sample: int = 64) -> bool:
    """True if most of the first *sample* non-empty values parse as floats."""
    seen = ok = 0
    for v in values:
        s = str(v).strip()
        if s == "":
            continue
        seen += 1
        try:
            float(s)
            ok += 1
        except ValueError:
            pass
        if seen >= sample:
            break
    return seen > 0 and (ok / seen) >= 0.8


def read_supplementary_matrix(
    source: Union[str, os.PathLike, io.IOBase],
    *,
    accession: Optional[str] = None,
    series_metadata: Optional[Dict[str, Any]] = None,
) -> GeoSeries:
    """Parse a supplementary expression matrix (counts / FPKM / TPM) offline.

    Handles the two shapes seen in the wild:

    * **Plain matrix** — a gene-id column plus one numeric column per sample
      (``gene\\tS1\\tS2 …``).  Every numeric, non-annotation column is a sample.
    * **Quantifier output** (Cufflinks/cuffdiff ``*_FPKM_Matrix``) — a block of
      annotation columns (``gene_id``, ``locus``, ``length`` …) followed by
      per-sample value columns interleaved with ``*_conf_lo``/``*_conf_hi``/
      ``*_status`` statistics.  The statistics and annotation are dropped; the
      remaining numeric columns (``Dex_FPKM``, ``Dex_LL14`` …) become samples.

    The delimiter (tab, comma, or run-of-spaces) is sniffed from the header.
    """
    src_path = None if hasattr(source, "read") else os.fspath(source)
    fh = _open_text(source)
    try:
        text = fh.read()
    finally:
        if not hasattr(source, "read"):
            fh.close()

    header_line = next(
        (ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")),
        "",
    )
    if not header_line:
        raise GeoError("supplementary matrix is empty")
    sep = _sniff_delimiter(header_line)

    # The fast C parser handles single-char delimiters; only the run-of-spaces
    # regex needs the (much slower) Python engine.  This matters a lot for the
    # large single-cell matrices that now pass the size cap.
    if sep == r"\s+":
        df = pd.read_csv(
            io.StringIO(text), sep=sep, engine="python",
            comment="#", dtype=str, keep_default_na=False,
        )
    else:
        df = pd.read_csv(
            io.StringIO(text), sep=sep, engine="c",
            comment="#", dtype=str, keep_default_na=False,
        )
    if df.shape[1] < 2:
        raise GeoError("supplementary file does not look like a matrix (<2 columns)")

    cols = [str(c) for c in df.columns]
    df.columns = cols
    low = {c: c.strip().lower() for c in cols}

    def _populated(col: str) -> float:
        vals = df[col].astype(str)
        n = len(vals)
        return float((vals.str.strip() != "").sum()) / n if n else 0.0

    # --- gene-id column -----------------------------------------------------
    # Walk the priority list, but skip a candidate that is sparsely populated
    # (annotated tables often carry a symbol column with blanks for novel/
    # unnamed features) in favour of a fully-populated id column further down.
    gene_col = None
    fallback_gene_col = None
    for cand in _GENE_COL_PRIORITY:
        for c in cols:
            if low[c] == cand:
                if fallback_gene_col is None:
                    fallback_gene_col = c
                if _populated(c) >= 0.95:
                    gene_col = c
                break
        if gene_col is not None:
            break
    if gene_col is None:
        gene_col = fallback_gene_col
    if gene_col is None:
        # Fall back to the first non-numeric column, else the first column.
        gene_col = next(
            (c for c in cols if not _mostly_numeric(df[c].tolist())), cols[0]
        )

    # --- sample / value columns --------------------------------------------
    # First pass: every numeric, non-annotation, non-statistic column.
    numeric_value_cols: List[str] = []
    for c in cols:
        if c == gene_col:
            continue
        name = low[c]
        if name in _ANNOT_NAME_DENY:
            continue
        if name.endswith(_ANNOT_SUFFIX_DENY):
            continue
        if not _mostly_numeric(df[c].tolist()):
            continue
        numeric_value_cols.append(c)

    # Split the surviving numeric columns into unit-tagged (grouped by unit)
    # and bare (un-suffixed) ones.
    families: Dict[str, List[str]] = {}
    unsuffixed: List[str] = []
    for c in numeric_value_cols:
        for tag in _EXPRESSION_SUFFIXES:
            if low[c].endswith(tag):
                families.setdefault(tag, []).append(c)
                break
        else:
            unsuffixed.append(c)

    if families and not unsuffixed:
        # Every sample column carries an explicit unit tag — the fingerprint of
        # an annotated quantifier / DESeq2 table that reports the SAME samples in
        # multiple units (``*_count`` and ``*_tpm``) plus per-condition aggregate
        # columns.  Trust one unit family (raw counts preferred over normalized)
        # so samples aren't double-counted and aggregates present in only one
        # unit fall away.
        value_cols = None
        for tag in _EXPRESSION_SUFFIXES:  # priority order: counts first
            cand = families.get(tag)
            if cand and len(cand) >= 2:
                value_cols = cand
                break
        if value_cols is None:
            value_cols = numeric_value_cols
    else:
        # Samples are (at least partly) un-suffixed — e.g. cuffdiff's ``Dex_FPKM``
        # condition means alongside bare ``Dex_R1`` replicates.  Both are real
        # samples, so trust the full numeric, non-statistic set.
        value_cols = numeric_value_cols

    if len(value_cols) < 2:
        raise GeoError(
            "supplementary matrix has fewer than 2 usable sample columns "
            f"(gene column {gene_col!r}; found {len(value_cols)})"
        )

    # Derive clean sample names: strip a trailing _FPKM/_TPM/_RPKM/_CPM tag so
    # cuffdiff's "Dex_FPKM" reads as sample "Dex"; de-duplicate collisions.
    def _clean(name: str) -> str:
        n = name
        for tag in _EXPRESSION_SUFFIXES:
            if n.lower().endswith(tag):
                return n[: -len(tag)]
        return n

    genes = df[gene_col].astype(str).map(lambda s: s.strip())
    data: Dict[str, List[float]] = {}
    used: Dict[str, int] = {}
    for c in value_cols:
        base = _clean(c) or c
        if base in used:
            used[base] += 1
            base = f"{base}.{used[base]}"
        else:
            used[base] = 0
        data[base] = [_to_float(v) for v in df[c].tolist()]

    expr = pd.DataFrame(data)
    expr.insert(0, GENE_COL, genes.values)
    expr = expr[expr[GENE_COL].astype(str).str.strip() != ""].reset_index(drop=True)
    if expr.empty:
        raise GeoError("supplementary matrix had no rows with a gene id")

    meta = dict(series_metadata or {})
    meta.setdefault("source", "supplementary")
    acc = (
        accession
        or _first(meta.get("geo_accession"))
        or _accession_from_path(src_path)
        or "<local>"
    )
    platform = _first(meta.get("platform_id"))
    return GeoSeries(
        accession=acc,
        expression=expr,
        samples=pd.DataFrame(),
        metadata=meta,
        platform=platform,
        source_path=src_path,
    )


def fetch_supplementary_series(
    accession: str,
    *,
    cache_dir: Optional[str] = None,
    force: bool = False,
    timeout: float = 60.0,
    series_metadata: Optional[Dict[str, Any]] = None,
) -> GeoSeries:
    """Find, download, and parse the best supplementary matrix for a series.

    Tries flat matrix-shaped files first (best name-ranked candidate first,
    content is the arbiter), then falls back to ``.tar`` archives — single-cell
    and microarray series frequently publish only a ``*_RAW.tar`` that either
    bundles the real matrix or holds one file per sample (see
    :func:`read_tar_matrix`).  Downloads are streamed with a size cap so a
    multi-GB matrix is skipped cleanly rather than hanging the pipeline.
    """
    acc = accession.strip().upper()
    names = list_supplementary_files(acc, timeout=timeout)
    if not names:
        raise GeoError(f"{acc}: no supplementary files published")
    ranked = rank_supplementary_matrices(names)
    tar_names = [n for n in names if n.lower().endswith(_TAR_EXTS)]
    if not ranked and not tar_names:
        raise GeoError(
            f"{acc}: no matrix-shaped supplementary file among {names!r}"
        )

    cdir = cache_dir or default_cache_dir()
    os.makedirs(cdir, exist_ok=True)
    max_bytes = _suppl_max_bytes()
    errors: List[str] = []

    def _fetch(choice: str) -> Optional[str]:
        dest = os.path.join(cdir, f"{acc}__{choice}")
        if force or not os.path.exists(dest) or os.path.getsize(dest) == 0:
            url = supplementary_dir_url(acc) + choice
            try:
                _download_capped(url, dest, timeout=timeout, max_bytes=max_bytes)
            except GeoError as exc:
                errors.append(f"{choice}: {exc}")
                return None
        return dest

    # 1) Flat, matrix-shaped files.  The file name only *ranks* candidates;
    # content is the arbiter, which rescues annotated/differential tables (e.g.
    # a DESeq2 ``.annot`` file whose ``*_count`` columns are the matrix).
    for choice in ranked:
        dest = _fetch(choice)
        if dest is None:
            continue
        meta = dict(series_metadata or {})
        meta["supplementary_file"] = choice
        meta["source"] = "supplementary"
        try:
            return read_supplementary_matrix(dest, accession=acc, series_metadata=meta)
        except GeoError as exc:
            errors.append(f"{choice}: {exc}")
            continue

    # 2) Archive fallback: a bundled matrix, or per-sample files to assemble.
    for tar in tar_names:
        dest = _fetch(tar)
        if dest is None:
            continue
        meta = dict(series_metadata or {})
        meta["supplementary_file"] = tar
        meta["source"] = "supplementary_tar"
        try:
            return read_tar_matrix(dest, accession=acc, series_metadata=meta)
        except GeoError as exc:
            errors.append(f"{tar}: {exc}")
            continue

    detail = "; ".join(errors) if errors else f"tried {ranked + tar_names!r}"
    raise GeoError(
        f"{acc}: no supplementary file yielded a usable expression matrix ({detail})"
    )


def read_tar_matrix(
    path: Union[str, os.PathLike],
    *,
    accession: Optional[str] = None,
    series_metadata: Optional[Dict[str, Any]] = None,
    member_cap: int = 8000,
) -> GeoSeries:
    """Parse an expression matrix out of a ``.tar`` / ``.tar.gz`` archive.

    Two archive shapes are handled, in order:

    1. **Bundled matrix** — one member is itself a matrix-shaped table
       (``*_counts.txt.gz`` …).  Members are name-ranked like flat supplementary
       files and the first that parses to a real matrix wins.
    2. **Per-sample files** — the archive holds one small table per sample (the
       classic single-cell / array ``*_RAW.tar``), each a gene-id column plus a
       single value column.  These are assembled column-wise into a matrix,
       keyed by a cleaned sample name derived from the member filename.
    """
    import tarfile

    try:
        tf = tarfile.open(path)
    except Exception as exc:  # noqa: BLE001
        raise GeoError(f"could not open tar archive {os.fspath(path)!r}: {exc}") from exc

    errors: List[str] = []
    try:
        members = [m for m in tf.getmembers() if m.isfile() and m.size > 0]
        if not members:
            raise GeoError("tar archive is empty")

        # (1) A member that is itself a full matrix.
        def _score(m: "tarfile.TarInfo") -> int:
            return _score_matrix_candidate(os.path.basename(m.name))

        bundled = sorted(
            (m for m in members if _score(m) > -100), key=_score, reverse=True
        )
        for m in bundled:
            tmp = None
            try:
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                # Extract to a temp file preserving the basename so the gzip
                # handling in read_supplementary_matrix kicks in for ``.gz``.
                suffix = "_" + os.path.basename(m.name)
                fd, tmp = tempfile.mkstemp(prefix="adgencov_tar_", suffix=suffix)
                with os.fdopen(fd, "wb") as out:
                    out.write(fobj.read())
                meta = dict(series_metadata or {})
                meta["archive_member"] = m.name
                return read_supplementary_matrix(tmp, accession=accession, series_metadata=meta)
            except GeoError as exc:
                errors.append(f"{m.name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{m.name}: {exc}")
            finally:
                if tmp is not None:
                    _rm_quiet(tmp)

        # (2) Per-sample assembly.
        try:
            return _assemble_per_sample_tar(
                tf, members, accession=accession,
                series_metadata=series_metadata, member_cap=member_cap,
            )
        except GeoError as exc:
            errors.append(f"per-sample assembly: {exc}")

        raise GeoError(
            "no usable matrix in archive ("
            + ("; ".join(errors) if errors else "no tabular members")
            + ")"
        )
    finally:
        tf.close()


def _sample_name_from_member(name: str) -> str:
    """Derive a sample label from an archive member path.

    ``GSM2154556_P1_A1.csv.gz`` → ``GSM2154556_P1_A1``.  Strips directory parts
    and known tabular/compression extensions.
    """
    base = os.path.basename(name)
    low = base.lower()
    for ext in (".tar.gz", ".tgz", ".tar"):
        if low.endswith(ext):
            base = base[: -len(ext)]
            low = base.lower()
    for ext in (".gz",):
        if low.endswith(ext):
            base = base[: -len(ext)]
            low = base.lower()
    for ext in (".txt", ".tsv", ".csv", ".tab"):
        if low.endswith(ext):
            base = base[: -len(ext)]
            break
    return base or name


def _assemble_per_sample_tar(
    tf: Any,
    members: Sequence[Any],
    *,
    accession: Optional[str],
    series_metadata: Optional[Dict[str, Any]],
    member_cap: int,
) -> GeoSeries:
    """Assemble a matrix from one-table-per-sample archive members.

    Each qualifying member must be a two-column (gene-id, value) table.  The
    per-member value columns become the samples; genes are aligned on the union
    of ids (missing entries filled with 0).  Requires ≥3 samples and ≥2 shared
    genes to be considered a real matrix.
    """
    tab = [
        m for m in members
        if os.path.basename(m.name).lower().endswith(_TABULAR_EXTS)
        and not os.path.basename(m.name).lower().startswith(".")
    ]
    if len(tab) < 3:
        raise GeoError(f"only {len(tab)} tabular members (need ≥3 per-sample files)")
    if len(tab) > member_cap:
        raise GeoError(
            f"{len(tab)} members exceeds the {member_cap}-file assembly cap"
        )

    columns: Dict[str, "pd.Series"] = {}
    used: Dict[str, int] = {}
    parsed = 0
    for m in tab:
        fobj = tf.extractfile(m)
        if fobj is None:
            continue
        raw = fobj.read()
        base = os.path.basename(m.name)
        if base.lower().endswith(".gz"):
            try:
                raw = gzip.decompress(raw)
            except OSError:
                continue
        text = raw.decode("utf-8", errors="replace")
        header_line = next(
            (ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")),
            "",
        )
        if not header_line:
            continue
        sep = _sniff_delimiter(header_line)
        try:
            sub = pd.read_csv(
                io.StringIO(text), sep=sep,
                engine="python" if sep == r"\s+" else "c",
                comment="#", dtype=str, keep_default_na=False, header=None,
            )
        except Exception:  # noqa: BLE001
            continue
        if sub.shape[1] < 2:
            continue
        # First column = gene ids; last mostly-numeric column = the value.
        gcol = 0
        vcol = None
        for c in range(sub.shape[1] - 1, 0, -1):
            if _mostly_numeric(sub[c].tolist()):
                vcol = c
                break
        if vcol is None:
            continue
        # Drop a header row if the value cell isn't numeric.
        first_val = str(sub.iloc[0, vcol]).strip()
        try:
            float(first_val)
        except ValueError:
            sub = sub.iloc[1:]
        genes = sub[gcol].astype(str).map(lambda s: s.strip())
        vals = [_to_float(v) for v in sub[vcol].tolist()]
        s = pd.Series(vals, index=genes.values)
        s = s[~s.index.duplicated(keep="first")]

        sample = _sample_name_from_member(m.name)
        if sample in used:
            used[sample] += 1
            sample = f"{sample}.{used[sample]}"
        else:
            used[sample] = 0
        columns[sample] = s
        parsed += 1

    if len(columns) < 3:
        raise GeoError(
            f"assembled only {len(columns)} usable per-sample columns (need ≥3)"
        )

    expr_num = pd.DataFrame(columns).fillna(0.0)
    if expr_num.shape[0] < 2:
        raise GeoError("assembled matrix has fewer than 2 shared genes")
    expr_num.index = [str(g).strip() for g in expr_num.index]
    expr = expr_num.reset_index().rename(columns={"index": GENE_COL})
    expr = expr[expr[GENE_COL].astype(str).str.strip() != ""].reset_index(drop=True)
    if expr.empty:
        raise GeoError("assembled matrix had no rows with a gene id")

    meta = dict(series_metadata or {})
    meta.setdefault("source", "supplementary_tar")
    meta["assembled_samples"] = len(columns)
    acc = (
        accession
        or _first(meta.get("geo_accession"))
        or "<local>"
    )
    return GeoSeries(
        accession=acc,
        expression=expr,
        samples=pd.DataFrame(),
        metadata=meta,
        platform=_first(meta.get("platform_id")),
        source_path=None,
    )


def _series_metadata_only(path: str) -> Dict[str, Any]:
    """Scan just the ``!Series_*`` header of a series matrix for metadata.

    Cheap best-effort so a supplementary-backed series still carries its title,
    summary, and platform even though its in-matrix table was empty.
    """
    meta: Dict[str, Any] = {}
    try:
        fh = _open_text(path)
    except OSError:
        return meta
    try:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line or line.startswith("!series_matrix_table_"):
                continue
            if line.startswith("!Series_"):
                key, _, rest = line.partition("\t")
                field_name = key[len("!Series_"):].strip()
                vals = [_strip_quotes(t) for t in rest.split("\t")] if rest else []
                _accumulate(meta, field_name, vals)
            elif line.startswith(("ID_REF", "!Sample_")):
                # Reached the sample block / table header — metadata is done.
                if line.startswith("ID_REF"):
                    break
    finally:
        fh.close()
    return {
        k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
        for k, v in meta.items()
    }


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
    progress: Optional[Any] = None,
    cv_folds: Optional[int] = None,
    criterion: str = "loo",
    ebic_gamma: float = 0.5,
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
    def report(fraction: float, phase: str) -> None:
        if progress is not None:
            progress(fraction, phase)

    acc_label = str(source) if _looks_like_accession(source) else "series"
    report(0.02, f"Fetching {acc_label} from GEO")
    series = load_series(source, cache_dir=cache_dir, force=force)
    if series.n_samples < 3:
        raise GeoError(
            f"series {series.accession!r} has {series.n_samples} samples; "
            "at least 3 are required for covariance estimation"
        )
    regex = series.sample_regex()

    # Materialize the TSV and drive the C++ loader so numbers match the CLI.
    report(0.10, f"Preprocessing {series.n_genes} genes → top {n_genes}")
    with tempfile.TemporaryDirectory(prefix="adgencov_geo_") as td:
        tsv = os.path.join(td, f"{series.accession}.tsv")
        series.write_tsv(tsv, gene_col="gene_short_name")
        data = load_expression_matrix(tsv, sample_regex=regex, gene_col="gene_short_name")
        dataset = preprocess(
            data, n_genes=n_genes, min_mean=min_mean, log_transform=log_transform
        )

    report(0.18, "Building gene blocks")
    labels = build_group_labels(dataset, group, n_blocks=n_blocks)
    codes = factorize(labels)
    X = np.asarray(dataset.X, dtype=float)

    # The estimator grid is the long pole; give it the [0.20, 0.98] band so the
    # bar keeps moving through leave-one-out scoring on many-sample series.
    def scaled(fraction: float, phase: str) -> None:
        report(0.20 + 0.78 * fraction, phase)

    result = analyze(
        X,
        codes,
        genes=list(dataset.genes),
        top_fraction=top_fraction,
        progress=scaled,
        cv_folds=cv_folds,
        criterion=criterion,
        ebic_gamma=ebic_gamma,
    )
    return result
