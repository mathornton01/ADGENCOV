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

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

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
    ebic_score,
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
    "ebic_score",
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


# The recommended estimator's full covariance matrix is shipped in the JSON
# payload so the dashboard/GUI can draw a heatmap.  A p-by-p matrix is
# O(p^2) numbers; above this dimension we omit it (the network/edge view still
# works) to keep the payload from ballooning past a few megabytes.
COVARIANCE_PAYLOAD_MAX_DIM = 600


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
        """A JSON-serializable summary (for FastAPI responses).

        Includes the recommended estimator's full covariance matrix (for the
        heatmap) when the gene count is small enough — see
        :data:`COVARIANCE_PAYLOAD_MAX_DIM`; otherwise ``covariance`` is ``None``
        and consumers fall back to the edge/network view.
        """
        p = len(self.genes)
        cov = np.asarray(self.best.covariance, dtype=float)
        covariance = cov.tolist() if p <= COVARIANCE_PAYLOAD_MAX_DIM else None
        return {
            "genes": self.genes,
            "labels": self.labels,
            "n_genes": p,
            "recommended": self.best.spec.method,
            "covariance": covariance,
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


@dataclass(frozen=True)
class _RankedEstimator:
    """A single scored candidate — the Python-loop analogue of the C++
    ``EstimatorResult``, exposing the same attributes :meth:`to_dict` reads.

    Built so :func:`analyze` can iterate the candidate grid one estimator at a
    time (reporting progress after each) while staying numerically identical to
    the C++ ``recommend_estimator``: it calls the very same ``loo_nll`` /
    ``estimate_covariance`` primitives and reproduces the condition-number and
    stable-sort logic verbatim.
    """

    spec: Any
    covariance: np.ndarray
    loo_nll: float
    condition_number: float


def _kfold_nll(
    X: np.ndarray, labels: List[int], spec: Any, k: int
) -> float:
    """k-fold cross-validated mean Gaussian NLL for one candidate.

    A faster alternative to the exact leave-one-out :func:`loo_nll`: instead of
    ``n`` refits it does ``k`` (contiguous, no-shuffle folds, matching
    ``sklearn.model_selection.KFold(shuffle=False)``).  Each held-out sample is
    scored exactly once under the Gaussian fit on its training fold, and the
    mean over all ``n`` samples is returned — same scale as ``loo_nll`` so the
    two are directly comparable, but ``n/k`` times fewer covariance estimations.

    Uses the same C++ primitives (:func:`estimate_covariance`,
    :func:`gaussian_nll_one`), both GIL-released, so it parallelizes identically.
    """
    n = X.shape[0]
    k = max(2, min(int(k), n))
    # Contiguous folds via array_split (handles n % k != 0 like sklearn KFold).
    fold_idx = np.array_split(np.arange(n), k)
    total = 0.0
    for te in fold_idx:
        if te.size == 0:
            continue
        mask = np.ones(n, dtype=bool)
        mask[te] = False
        Xtr = X[mask]
        if Xtr.shape[0] < 3:  # need >=3 rows for an unbiased training covariance
            raise ValueError("k-fold: training fold too small")
        mu = Xtr.mean(axis=0)
        Sigma = estimate_covariance(Xtr, labels, spec)
        for i in te:
            total += gaussian_nll_one(X[i], mu, Sigma)
    return total / n


#: The selection criteria :func:`analyze` (and the API/UI) accept.  All three
#: are on a "lower is better" scale, so they substitute for one another directly.
CRITERIA = ("loo", "ebic", "kfold")


def _resolve_criterion(
    criterion: Optional[str], cv_folds: Optional[int]
) -> str:
    """Map the public (criterion, cv_folds) knobs onto one of :data:`CRITERIA`.

    ``criterion`` takes precedence when given.  For backward compatibility a bare
    ``cv_folds=k`` (with the default ``criterion``) still selects k-fold CV, so
    existing callers keep their behaviour unchanged.
    """
    crit = (criterion or "loo").lower()
    if crit not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    if crit == "loo" and cv_folds is not None:
        return "kfold"
    return crit


def _score_one(
    X: np.ndarray,
    labels: List[int],
    spec: Any,
    cv_folds: Optional[int] = None,
    criterion: str = "loo",
    ebic_gamma: float = 0.5,
) -> Optional[_RankedEstimator]:
    """Score a single candidate; return ``None`` if it fails to estimate.

    The heavy calls (:func:`loo_nll` / :func:`_kfold_nll` / :func:`ebic_score`,
    :func:`estimate_covariance`) run in the C++ core with the GIL released, so
    calling this from a thread pool yields genuine multi-core parallelism.  The
    ``criterion`` picks the scoring rule: ``"loo"`` (exact leave-one-out CV,
    default), ``"kfold"`` (k-fold CV with ``cv_folds`` folds, default 5), or
    ``"ebic"`` (one-pass Extended BIC with penalty ``ebic_gamma``).
    """
    try:
        crit = _resolve_criterion(criterion, cv_folds)
        if crit == "ebic":
            score = ebic_score(X, labels, spec, ebic_gamma)
        elif crit == "kfold":
            score = _kfold_nll(X, labels, spec, cv_folds if cv_folds is not None else 5)
        else:
            score = loo_nll(X, labels, spec)
        Sigma = np.asarray(estimate_covariance(X, labels, spec), dtype=float)
        # 2-norm condition number of an SPD matrix = |lambda|_max / |lambda|_min.
        ev = np.abs(np.linalg.eigvalsh(Sigma))
        lo = float(ev.min())
        cond = (float(ev.max()) / lo) if lo > 0.0 else float("inf")
        return _RankedEstimator(spec, Sigma, float(score), cond)
    except Exception:  # noqa: BLE001 - skip candidates that fail to estimate
        return None


# Cap the recommender's worker pool.  Scoring is CPU-bound in released-GIL C++,
# so more workers than cores just adds memory pressure (each fold allocates
# p-by-p temporaries).  Override with ADGENCOV_MAX_WORKERS for tuning.
def _cpu_budget() -> int:
    """CPUs this process may actually use — container quota aware.

    ``os.cpu_count()`` reports the *host's* cores and ignores cgroup CPU limits,
    so inside a small container (Railway, Kubernetes, ECS) it over-subscribes
    wildly: the recommender would start one CPU-bound thread per host core on a
    1-2 core slice, saturate it, and starve everything else in the process —
    including the HTTP server serving /health, which then looks like an outage.
    """
    # Start from affinity (respects taskset/cpuset), then apply the cgroup quota
    # below. Affinity alone is NOT enough: container runtimes such as Railway
    # cap CPU with a cfs quota while leaving affinity set to every host core, so
    # returning here would report the host's cores and defeat the whole point.
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        n = 0
    if n <= 0:
        n = os.cpu_count() or 1

    # cgroup v2, then v1: quota/period == the fractional core allowance.
    for quota_path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            with open(quota_path) as fh:
                raw = fh.read().split()
            if period_path is None:                 # v2: "<quota|max> <period>"
                if raw[0] == "max":
                    continue
                quota, period = int(raw[0]), int(raw[1])
            else:                                   # v1: separate files
                quota = int(raw[0])
                with open(period_path) as fh:
                    period = int(fh.read().strip())
            if quota > 0 and period > 0:
                n = min(n, max(1, quota // period))
        except (OSError, ValueError, IndexError):
            continue
    return max(1, n)


def _max_workers(n_tasks: int) -> int:
    env = os.environ.get("ADGENCOV_MAX_WORKERS")
    if env:
        try:
            cap = int(env)
            if cap > 0:
                return min(n_tasks, cap)
        except ValueError:
            pass
    budget = _cpu_budget()
    # Leave one CPU for the rest of the process (the API event loop and its
    # thread pool) so a running analysis can never make the service unreachable.
    if budget > 1:
        budget -= 1
    return max(1, min(n_tasks, budget))


def _rank_estimators(
    X: np.ndarray,
    labels: List[int],
    progress: Optional[Callable[[float, str], None]] = None,
    cv_folds: Optional[int] = None,
    criterion: str = "loo",
    ebic_gamma: float = 0.5,
) -> List[_RankedEstimator]:
    """Score the candidate grid, ascending by CV NLL, reporting per-candidate
    progress.  With ``cv_folds=None`` (default) this is numerically identical to
    ``adgencov::recommend_estimator`` — same primitives, same stable sort — but
    scores the grid concurrently across cores (the C++ calls release the GIL)
    and reports progress as each candidate finishes.  With ``cv_folds=k`` it
    scores by k-fold CV instead of exact leave-one-out (faster; scores shift)."""
    grid = candidate_grid(X.shape[1], X.shape[0])
    total = len(grid)
    # Slot results by grid index so ties keep candidate-grid order under the
    # stable sort below, exactly as the sequential C++ path does.
    scored: List[Optional[_RankedEstimator]] = [None] * total

    done = 0
    lock = threading.Lock()
    if progress is not None:
        progress(0.0, f"Scoring {total} estimators")

    with ThreadPoolExecutor(max_workers=_max_workers(total)) as pool:
        futures = {
            pool.submit(
                _score_one, X, labels, spec, cv_folds, criterion, ebic_gamma
            ): i
            for i, spec in enumerate(grid)
        }
        for fut, idx in _as_completed_pairs(futures):
            scored[idx] = fut.result()
            if progress is not None:
                with lock:
                    done += 1
                    progress(done / total, f"Scored {done}/{total} estimators")
    if progress is not None:
        progress(1.0, "Extracting covariance network")

    results = [r for r in scored if r is not None]
    # Stable ascending sort by LOO-NLL (Python's sort is stable), matching the
    # C++ std::stable_sort so ties keep candidate-grid order.
    results.sort(key=lambda r: r.loo_nll)
    return results


def _as_completed_pairs(futures: Dict[Any, int]):
    """Yield ``(future, index)`` as each future completes."""
    from concurrent.futures import as_completed

    for fut in as_completed(futures):
        yield fut, futures[fut]


def analyze(
    X: np.ndarray,
    labels: Sequence[int],
    genes: Optional[Sequence[str]] = None,
    top_fraction: float = 0.01,
    progress: Optional[Callable[[float, str], None]] = None,
    cv_folds: Optional[int] = None,
    criterion: str = "loo",
    ebic_gamma: float = 0.5,
) -> AnalysisResult:
    """Run the recommender end-to-end on a standardized samples-by-genes matrix.

    This is the one call the backend makes after preprocessing/grouping: it
    ranks the estimator grid by cross-validated NLL and extracts the top
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
    progress : callable, optional
        ``progress(fraction, phase)`` invoked after each candidate is scored so
        a UI can show a live status bar.  When ``None``, scoring is unobserved.
    cv_folds : int, optional
        When ``None`` (default) candidates are scored by exact leave-one-out CV,
        numerically identical to the C++ ``recommend_estimator``.  When set to an
        integer ``k``, k-fold CV is used instead: ``n/k`` times fewer covariance
        fits, so much faster on large sample counts, at the cost of a slightly
        different (higher-variance) score.  The recommended estimator is usually
        unchanged; use it to trade a little selection precision for speed.
    criterion : {"loo", "ebic", "kfold"}, optional
        Model-selection rule for ranking the estimator grid.  ``"loo"`` (default)
        is exact leave-one-out CV; ``"kfold"`` is k-fold CV (``cv_folds`` folds,
        default 5); ``"ebic"`` is one-pass Extended BIC — ``~n`` times cheaper
        than leave-one-out — with the ``ebic_gamma`` high-dimensional penalty.
        A bare ``cv_folds=k`` still implies k-fold for backward compatibility.
    ebic_gamma : float, optional
        Extended BIC penalty in ``[0, 1]`` used only when ``criterion="ebic"``
        (``0`` = ordinary BIC; larger is more conservative for ``p >> n``).

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
    ranking = _rank_estimators(
        X,
        labels,
        progress=progress,
        cv_folds=cv_folds,
        criterion=criterion,
        ebic_gamma=ebic_gamma,
    )
    if not ranking:
        raise RuntimeError("no estimator in the grid produced a valid fit")
    edges = top_edges(np.asarray(ranking[0].covariance), genes, top_fraction)
    return AnalysisResult(genes=genes, labels=labels, ranking=ranking, top_edges=edges)
