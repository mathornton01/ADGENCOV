"""Pipeline runners that turn a request into a JSON analysis payload.

Two entry points, one for each source, both returning
``AnalysisResult.to_dict()``:

* :func:`run_upload_analysis` — an uploaded expression matrix (bytes on disk),
  driven through the same steps the CLI uses so the numbers match exactly:
  ``load_expression_matrix → preprocess → build_group_labels → factorize →
  analyze``.
* :func:`run_geo_analysis` — a GEO accession (or local series-matrix path),
  delegating to :func:`adgencov.geo.analyze_series`.

Keeping these here (not in the FastAPI handlers) makes them trivially unit- and
job-testable without HTTP.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, Optional

import numpy as np

# progress(fraction in [0, 1], phase label) — optional live status callback.
ProgressFn = Callable[[float, str], None]


def _noop_progress(fraction: float, phase: str) -> None:
    """Default progress sink when a caller doesn't supply one."""

from .. import analyze
from .._core import (
    build_group_labels,
    factorize,
    load_expression_matrix,
    preprocess,
)
from .models import GeoAnalyzeRequest, MultiAnalyzeRequest, UploadParams
from ..pipeline import choose_group, grouping_meta


def run_upload_analysis(
    raw: bytes,
    params: UploadParams,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Analyze an uploaded matrix file and return the JSON-serializable result.

    *raw* is the raw file bytes (TSV/CSV/whitespace — the loader sniffs the
    delimiter).  The bytes are written to a temp file because the C++ loader
    reads from a path; the temp file is always cleaned up.  *progress* is an
    optional ``progress(fraction, phase)`` callback driving the status bar.
    """
    report = progress or _noop_progress
    if not raw.strip():
        raise ValueError("uploaded file is empty")

    report(0.02, "Reading uploaded matrix")
    fd, path = tempfile.mkstemp(prefix="adgencov_upload_", suffix=".tsv")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        data = load_expression_matrix(
            path, sample_regex=params.sample_regex, gene_col=params.gene_col
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    report(0.10, f"Preprocessing → top {params.n_genes} genes")
    dataset = preprocess(
        data,
        n_genes=params.n_genes,
        min_mean=params.min_mean,
        log_transform=params.log_transform,
    )
    n_samples = np.asarray(dataset.X, dtype=float).shape[0]
    if n_samples < 3:
        raise ValueError(
            f"matrix has {n_samples} samples after preprocessing; "
            "at least 3 are required for covariance estimation"
        )

    X = np.asarray(dataset.X, dtype=float)
    genes = list(dataset.genes)
    sel = dict(
        top_fraction=params.top_fraction,
        cv_folds=params.cv_folds,
        criterion=params.criterion,
        ebic_gamma=params.ebic_gamma,
        families=params.families,
        ad_modes=params.ad_modes,
        sweep=params.sweep,
    )

    def run_one(group, n_blocks, prog):
        labels = build_group_labels(dataset, group, n_blocks=n_blocks)
        codes = factorize(labels)
        return analyze(X, codes, genes=genes, progress=prog, **sel)

    if params.group == "auto":
        report(0.18, "Trying built-in symmetry structures")
        result, gmeta = choose_group(
            run_one, default_n_blocks=params.n_blocks,
            progress=_band(report, 0.20, 0.98),
        )
    else:
        report(0.18, "Building gene blocks")
        result = run_one(params.group, params.n_blocks, _band(report, 0.20, 0.98))
        gmeta = grouping_meta(params.group, params.n_blocks)

    payload = result.to_dict()
    payload["source"] = {"kind": "upload"}
    payload["grouping"] = gmeta
    return payload


def run_geo_analysis(
    req: GeoAnalyzeRequest,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Fetch/parse a GEO series and return the JSON-serializable result."""
    # Imported lazily so the service module stays importable without pandas
    # (the API package itself only hard-depends on the core + numpy).
    from ..geo import analyze_series

    from ..geo import load_series

    report = progress or _noop_progress
    sel = dict(
        n_genes=req.n_genes, min_mean=req.min_mean, log_transform=req.log_transform,
        top_fraction=req.top_fraction, cv_folds=req.cv_folds, criterion=req.criterion,
        ebic_gamma=req.ebic_gamma, families=req.families, ad_modes=req.ad_modes,
        sweep=req.sweep,
    )

    if req.group == "auto":
        # Fetch/parse once, then reuse the parsed series for every grouping so a
        # group sweep does not re-download the accession.
        report(0.02, f"Fetching {req.accession} from GEO")
        series = load_series(req.accession, force=req.force)

        def run_one(group, n_blocks, prog):
            return analyze_series(series, group=group, n_blocks=n_blocks,
                                  progress=prog, **sel)

        result, gmeta = choose_group(
            run_one, default_n_blocks=req.n_blocks, progress=_band(report, 0.05, 0.99),
        )
    else:
        result = analyze_series(
            req.accession, group=req.group, n_blocks=req.n_blocks,
            force=req.force, progress=report, **sel,
        )
        gmeta = grouping_meta(req.group, req.n_blocks)

    payload = result.to_dict()
    payload["source"] = {"kind": "geo", "accession": req.accession}
    payload["grouping"] = gmeta
    return payload


def run_combine_analysis(
    req: "MultiAnalyzeRequest",
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Pool several GEO series into one matrix and analyze them together."""
    from ..multi import combine_series

    return combine_series(
        list(req.accessions),
        n_genes=req.n_genes,
        min_mean=req.min_mean,
        log_transform=req.log_transform,
        group=req.group,
        n_blocks=req.n_blocks,
        top_fraction=req.top_fraction,
        criterion=req.criterion,
        ebic_gamma=req.ebic_gamma,
        cv_folds=req.cv_folds,
        force=req.force,
        progress=progress or _noop_progress,
    )


def run_compare_analysis(
    req: "MultiAnalyzeRequest",
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Analyze several GEO series separately and compare the recovered structure."""
    from ..multi import compare_series

    return compare_series(
        list(req.accessions),
        n_genes=req.n_genes,
        min_mean=req.min_mean,
        log_transform=req.log_transform,
        group=req.group,
        n_blocks=req.n_blocks,
        top_fraction=req.top_fraction,
        criterion=req.criterion,
        ebic_gamma=req.ebic_gamma,
        cv_folds=req.cv_folds,
        force=req.force,
        progress=progress or _noop_progress,
    )


def _band(report: ProgressFn, lo: float, hi: float) -> ProgressFn:
    """Map a child stage's [0, 1] progress into the [lo, hi] slice of the whole."""

    def scaled(fraction: float, phase: str) -> None:
        report(lo + (hi - lo) * fraction, phase)

    return scaled
