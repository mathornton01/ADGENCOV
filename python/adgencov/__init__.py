"""ADGENCOV — Algebraic-Diversity Genetic Covariance for transcriptomics.

This is the friendly Python API over the compiled C++ core (``adgencov._core``,
built by pybind11).  Everything numerically heavy — the Reynolds/commutant
projection, the estimator family, leave-one-out model selection, and
agglomerative clustering — runs in vectorised Eigen; this layer adds a small
amount of orchestration glue and a single high-level :func:`analyze` entry
point that the FastAPI service and the web/desktop GUIs call.

The design boundary: Python owns everything network- and parsing-heavy (GEO
ingestion, database comparison, HTTP), C++ owns the numerics.  See the roadmap
in README.md for the phased plan (this module is Phase A).

Example
-------
>>> import numpy as np, adgencov
>>> X = np.random.default_rng(0).standard_normal((20, 8))
>>> labels = [0, 0, 1, 1, 1, 2, 2, 2]
>>> result = adgencov.analyze(X, labels)
>>> result.best.spec.method            # doctest: +SKIP
'ad_oas'
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# The compiled extension is placed next to this file by the CMake build
# (bindings/CMakeLists.txt sets LIBRARY_OUTPUT_DIRECTORY to python/adgencov/).
from . import _core  # noqa: F401  (re-exported below)

# Re-export the whole C++ surface so callers can reach it directly if they want.
from ._core import (  # noqa: F401
    EstimatorResult,
    EstimatorSpec,
    agglomerative_average,
    build_group_labels,
    candidate_grid,
    estimate_covariance,
    factorize,
    gaussian_nll_one,
    gene_family_label,
    ledoit_wolf,
    ledoit_wolf_shrinkage,
    load_expression_matrix,
    loo_nll,
    make_pd,
    oas,
    oas_shrinkage,
    preprocess,
    read_table,
    recommend_estimator,
    reynolds_project,
    ridge,
    sample_covariance,
    soft_threshold_offdiag,
)

__version__ = _core.__version__
__all__ = [
    "__version__",
    "analyze",
    "AnalysisResult",
    "Edge",
    "top_edges",
    # C++ core re-exports:
    "EstimatorResult",
    "EstimatorSpec",
    "agglomerative_average",
    "build_group_labels",
    "candidate_grid",
    "estimate_covariance",
    "factorize",
    "gaussian_nll_one",
    "gene_family_label",
    "ledoit_wolf",
    "ledoit_wolf_shrinkage",
    "load_expression_matrix",
    "loo_nll",
    "make_pd",
    "oas",
    "oas_shrinkage",
    "preprocess",
    "read_table",
    "recommend_estimator",
    "reynolds_project",
    "ridge",
    "sample_covariance",
    "soft_threshold_offdiag",
]


@dataclass(frozen=True)
class Edge:
    """One covariance edge between two genes (top-fraction network)."""

    gene_a: str
    gene_b: str
    covariance: float
    abs_covariance: float


def top_edges(
    Sigma: np.ndarray,
    genes: Sequence[str],
    top_fraction: float = 0.01,
) -> List[Edge]:
    """Return the strongest covariance edges, mirroring the prototype's ``top_edges``.

    Ranks the ``p*(p-1)/2`` upper-triangular entries of ``Sigma`` by absolute
    value and keeps the top ``round(top_fraction * n_pairs)`` (at least one).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    p = Sigma.shape[0]
    if len(genes) != p:
        raise ValueError(f"genes length {len(genes)} != Sigma dim {p}")
    iu, ju = np.triu_indices(p, k=1)
    vals = Sigma[iu, ju]
    order = np.argsort(-np.abs(vals), kind="stable")
    n_pairs = len(order)
    k = max(1, int(round(top_fraction * n_pairs)))
    edges: List[Edge] = []
    for idx in order[:k]:
        i, j = int(iu[idx]), int(ju[idx])
        c = float(Sigma[i, j])
        edges.append(Edge(genes[i], genes[j], c, abs(c)))
    return edges


@dataclass(frozen=True)
class AnalysisResult:
    """The full result of :func:`analyze` — ready to serialize for the API/GUI."""

    genes: List[str]
    labels: List[int]
    ranking: List[EstimatorResult]
    top_edges: List[Edge]

    @property
    def best(self) -> EstimatorResult:
        """The recommended (lowest-LOO-NLL) estimator."""
        return self.ranking[0]

    @property
    def covariance(self) -> np.ndarray:
        """The best estimator's SPD covariance matrix."""
        return np.asarray(self.best.covariance)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable summary (for FastAPI responses)."""
        return {
            "genes": self.genes,
            "labels": self.labels,
            "recommended": self.best.spec.method,
            "ranking": [
                {
                    "method": r.spec.method,
                    "params": dict(r.spec.params),
                    "loo_nll": r.loo_nll,
                    "condition_number": r.condition_number,
                }
                for r in self.ranking
            ],
            "edges": [
                {
                    "gene_a": e.gene_a,
                    "gene_b": e.gene_b,
                    "covariance": e.covariance,
                    "abs_covariance": e.abs_covariance,
                }
                for e in self.top_edges
            ],
        }


def analyze(
    X: np.ndarray,
    labels: Sequence[int],
    genes: Optional[Sequence[str]] = None,
    top_fraction: float = 0.01,
) -> AnalysisResult:
    """Run the recommender end-to-end on a standardized samples-by-genes matrix.

    This is the one call the backend makes after preprocessing/grouping: it
    ranks the estimator grid by leave-one-out NLL and extracts the top
    covariance edges from the winner.

    Parameters
    ----------
    X : (n, p) array
        Standardized samples-by-genes matrix (e.g. from :func:`preprocess`).
    labels : sequence of int, length p
        Integer block/group id per gene (e.g. from :func:`factorize`).
    genes : sequence of str, optional
        Gene names for edge labels; defaults to ``gene_0 ... gene_{p-1}``.
    top_fraction : float
        Fraction of gene pairs to keep as edges.

    Returns
    -------
    AnalysisResult
    """
    X = np.asarray(X, dtype=float)
    labels = [int(v) for v in labels]
    p = X.shape[1]
    if genes is None:
        genes = [f"gene_{i}" for i in range(p)]
    else:
        genes = list(genes)
    ranking = recommend_estimator(X, labels)
    if not ranking:
        raise RuntimeError("no estimator in the grid produced a valid fit")
    edges = top_edges(np.asarray(ranking[0].covariance), genes, top_fraction)
    return AnalysisResult(genes=genes, labels=labels, ranking=ranking, top_edges=edges)
