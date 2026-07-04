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
from typing import Any, Dict

import numpy as np

from .. import analyze
from .._core import (
    build_group_labels,
    factorize,
    load_expression_matrix,
    preprocess,
)
from .models import GeoAnalyzeRequest, UploadParams


def run_upload_analysis(raw: bytes, params: UploadParams) -> Dict[str, Any]:
    """Analyze an uploaded matrix file and return the JSON-serializable result.

    *raw* is the raw file bytes (TSV/CSV/whitespace — the loader sniffs the
    delimiter).  The bytes are written to a temp file because the C++ loader
    reads from a path; the temp file is always cleaned up.
    """
    if not raw.strip():
        raise ValueError("uploaded file is empty")

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

    labels = build_group_labels(dataset, params.group, n_blocks=params.n_blocks)
    codes = factorize(labels)
    X = np.asarray(dataset.X, dtype=float)
    result = analyze(X, codes, genes=list(dataset.genes), top_fraction=params.top_fraction)
    payload = result.to_dict()
    payload["source"] = {"kind": "upload"}
    return payload


def run_geo_analysis(req: GeoAnalyzeRequest) -> Dict[str, Any]:
    """Fetch/parse a GEO series and return the JSON-serializable result."""
    # Imported lazily so the service module stays importable without pandas
    # (the API package itself only hard-depends on the core + numpy).
    from ..geo import analyze_series

    result = analyze_series(
        req.accession,
        n_genes=req.n_genes,
        min_mean=req.min_mean,
        log_transform=req.log_transform,
        group=req.group,
        n_blocks=req.n_blocks,
        top_fraction=req.top_fraction,
        force=req.force,
    )
    payload = result.to_dict()
    payload["source"] = {"kind": "geo", "accession": req.accession}
    return payload
