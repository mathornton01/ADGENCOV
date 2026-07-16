"""Multi-dataset analysis: combine several GEO series, or compare them.

Two complementary entry points over :mod:`adgencov.geo`:

* :func:`combine_series` merges several series into a single matrix and runs one
  covariance analysis on the pooled samples.  This directly attacks the ``p >> n``
  problem the method targets: each individual transcriptomic study is
  sample-starved, but several studies of the same tissue together are less so.
* :func:`compare_series` analyzes each series independently under identical
  settings and reports how far the recovered structure agrees — which estimator
  each dataset selects, and how much the top-covariance edge sets overlap.

Combining expression studies naively is not safe: different series use different
platforms, library sizes, and units, so raw values are not comparable across
datasets.  :func:`combine_series` therefore

1. reduces every series to the genes shared by all of them,
2. ranks genes by their mean *within-dataset* variance (a gene must vary inside
   the studies, not merely differ between them),
3. standardizes each gene *within each dataset* before pooling, which removes
   per-dataset location and scale — the dominant batch effect — so the pooled
   covariance reflects within-study co-variation rather than study identity.

Step 3 is why selection in step 2 uses within-dataset variance: after
standardization every gene has unit variance in every dataset, so a
variance ranking computed on the pooled matrix would be degenerate.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import analyze
from ._core import Dataset, build_group_labels, factorize

ProgressFn = Callable[[float, str], None]


def _noop(fraction: float, phase: str) -> None:
    """Default progress sink."""


class MultiError(RuntimeError):
    """Raised when several datasets cannot be combined or compared."""


# ---------------------------------------------------------------------------
# per-series preparation (shared by combine + compare)
# ---------------------------------------------------------------------------
def _series_matrix(series, min_mean: float, log_transform: bool):
    """Return (genes, L) for one series: genes-by-samples, deduped/filtered/logged.

    Deliberately stops short of standardizing: :func:`combine_series` needs the
    unstandardized values to rank genes and to z-score per dataset itself.
    """
    from .geo import GENE_COL

    df = series.expression
    ids = series.sample_ids
    if len(ids) < 2:
        raise MultiError(f"{series.accession}: only {len(ids)} sample(s)")
    genes = df[GENE_COL].astype(str).to_numpy()
    V = np.asarray(df[ids].to_numpy(), dtype=float)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.clip(V, 0.0, None)

    # Collapse duplicate symbols, keeping the highest-mean row (as preprocess does).
    means = V.mean(axis=1)
    order = np.argsort(-means)
    seen: Dict[str, None] = {}
    keep: List[int] = []
    for i in order:
        g = genes[i]
        if g not in seen:
            seen[g] = None
            keep.append(int(i))
    keep_idx = np.sort(np.asarray(keep, dtype=int))
    genes, V = genes[keep_idx], V[keep_idx]

    sel = V.mean(axis=1) >= float(min_mean)
    genes, V = genes[sel], V[sel]
    if genes.size == 0:
        raise MultiError(f"{series.accession}: no genes survived min_mean={min_mean}")
    L = np.log2(V + 1.0) if log_transform else V
    return genes, L


def _zscore_rows(L: np.ndarray) -> np.ndarray:
    """Standardize each gene (row) within one dataset; sd floored like the core."""
    mu = L.mean(axis=1, keepdims=True)
    sd = L.std(axis=1, ddof=1, keepdims=True)
    sd = np.maximum(sd, 1e-7)
    return (L - mu) / sd


def _load_all(accessions: Sequence[Union[str, Any]], *, cache_dir, force, report, lo, hi):
    from .geo import load_series

    out = []
    n = max(1, len(accessions))
    for i, acc in enumerate(accessions):
        label = acc if isinstance(acc, str) else getattr(acc, "accession", "series")
        report(lo + (hi - lo) * (i / n), f"Fetching {label}")
        out.append(load_series(acc, cache_dir=cache_dir, force=force))
    return out


def _common_panel(prepared, n_genes: int):
    """Shared gene panel across datasets, ranked by mean within-dataset variance.

    Returns ``(top_genes, per_ds)`` where *per_ds* is a list of
    ``(series, L_panel)`` with ``L_panel`` the dataset's unstandardized
    genes-by-samples values restricted to *top_genes*, in a common row order.

    Both entry points need this: :func:`combine_series` pools the panel, and
    :func:`compare_series` analyzes each dataset *over the same panel* — without
    a common panel each dataset would select its own top-variance genes, the
    gene sets would barely intersect, and edge overlap would be trivially zero.
    """
    common = set(prepared[0][1])
    for _, genes, _ in prepared[1:]:
        common &= set(genes)
    if len(common) < 4:
        raise MultiError(
            f"only {len(common)} gene(s) are shared by all {len(prepared)} datasets; "
            "they may use different identifier spaces (e.g. probe ids vs symbols)"
        )

    common_sorted = sorted(common)
    idx = {g: i for i, g in enumerate(common_sorted)}
    var_acc = np.zeros(len(common_sorted), dtype=float)
    aligned = []
    for s, genes, L in prepared:
        rows = np.array([i for i, g in enumerate(genes) if g in idx], dtype=int)
        sub_genes, sub_L = genes[rows], L[rows]
        order = np.argsort([idx[g] for g in sub_genes])
        sub_L = sub_L[order]                       # aligned to common_sorted
        var_acc += sub_L.var(axis=1)
        aligned.append((s, sub_L))
    var_acc /= len(aligned)

    k = max(2, min(int(n_genes), len(common_sorted)))
    top = np.argsort(-var_acc)[:k]
    top_genes = [common_sorted[i] for i in top]
    return top_genes, [(s, L[top]) for s, L in aligned], len(common_sorted)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------
def combine_series(
    accessions: Sequence[Union[str, Any]],
    *,
    n_genes: int = 150,
    min_mean: float = 0.1,
    log_transform: bool = True,
    group: str = "gene_family",
    n_blocks: int = 4,
    top_fraction: float = 0.01,
    criterion: str = "loo",
    ebic_gamma: float = 0.5,
    cv_folds: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force: bool = False,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Pool several series into one matrix and run a single covariance analysis.

    Returns the usual ``AnalysisResult.to_dict()`` payload with an extra
    ``combined`` block describing what was merged (per-dataset sample counts and
    the size of the shared gene space).
    """
    report = progress or _noop
    if len(accessions) < 2:
        raise MultiError("combine needs at least two datasets")

    series_list = _load_all(
        accessions, cache_dir=cache_dir, force=force, report=report, lo=0.02, hi=0.35
    )

    report(0.38, "Harmonizing genes across datasets")
    prepared = [(s, *_series_matrix(s, min_mean, log_transform)) for s in series_list]
    top_genes, per_ds, n_shared = _common_panel(prepared, n_genes)
    k = len(top_genes)

    report(0.45, f"Pooling {len(per_ds)} datasets over {k} shared genes")
    blocks, contrib = [], []
    for s, L_panel in per_ds:
        Z = _zscore_rows(L_panel)                 # standardize within dataset
        blocks.append(Z.T)                        # samples x genes
        contrib.append({"accession": s.accession, "n_samples": int(Z.shape[1]),
                        "title": s.title, "platform": s.platform})
    X = np.vstack(blocks)
    if X.shape[0] < 3:
        raise MultiError(f"pooled matrix has only {X.shape[0]} samples")

    report(0.5, "Building gene blocks")
    labels = build_group_labels(Dataset(X, list(top_genes)), group, n_blocks=n_blocks)
    codes = factorize(labels)

    result = analyze(
        X, codes, genes=list(top_genes), top_fraction=top_fraction,
        progress=_band(report, 0.52, 0.98), cv_folds=cv_folds,
        criterion=criterion, ebic_gamma=ebic_gamma,
    )
    payload = result.to_dict()
    payload["source"] = {"kind": "combine",
                         "accessions": [c["accession"] for c in contrib]}
    payload["combined"] = {
        "datasets": contrib,
        "n_datasets": len(contrib),
        "n_samples_total": int(X.shape[0]),
        "n_shared_genes": n_shared,
        "n_genes_analyzed": k,
        "batch_control": "per-gene z-score within each dataset before pooling",
    }
    return payload


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def _edge_key(e: Dict[str, Any]) -> Tuple[str, str]:
    a, b = e["gene_a"], e["gene_b"]
    return (a, b) if a <= b else (b, a)


def compare_series(
    accessions: Sequence[Union[str, Any]],
    *,
    n_genes: int = 150,
    min_mean: float = 0.1,
    log_transform: bool = True,
    group: str = "gene_family",
    n_blocks: int = 4,
    top_fraction: float = 0.01,
    criterion: str = "loo",
    ebic_gamma: float = 0.5,
    cv_folds: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force: bool = False,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Analyze each series separately and report how far they agree.

    Every dataset is run through the identical pipeline, then compared on
    (a) which estimator each one selects, (b) how much their top-edge sets
    overlap (Jaccard), and (c) whether shared edges agree in sign.
    """
    report = progress or _noop
    if len(accessions) < 2:
        raise MultiError("compare needs at least two datasets")

    series_list = _load_all(
        accessions, cache_dir=cache_dir, force=force, report=report, lo=0.02, hi=0.30
    )
    report(0.33, "Harmonizing genes across datasets")
    prepared = [(s, *_series_matrix(s, min_mean, log_transform)) for s in series_list]
    # Every dataset is analyzed over the SAME gene panel, so the resulting
    # networks are directly comparable (see _common_panel).
    top_genes, per_ds, n_shared = _common_panel(prepared, n_genes)

    n = len(per_ds)
    per: List[Dict[str, Any]] = []
    for i, (s, L_panel) in enumerate(per_ds):
        lo, hi = 0.35 + 0.6 * (i / n), 0.35 + 0.6 * ((i + 1) / n)
        report(lo, f"Analyzing {s.accession} ({i + 1}/{n})")
        X = _zscore_rows(L_panel).T                 # samples x genes
        if X.shape[0] < 3:
            raise MultiError(f"{s.accession}: only {X.shape[0]} samples after preprocessing")
        labels = build_group_labels(Dataset(X, list(top_genes)), group, n_blocks=n_blocks)
        codes = factorize(labels)
        d = analyze(
            X, codes, genes=list(top_genes), top_fraction=top_fraction,
            progress=_band(report, lo, hi), cv_folds=cv_folds,
            criterion=criterion, ebic_gamma=ebic_gamma,
        ).to_dict()
        per.append({
            "accession": s.accession,
            "title": s.title,
            "recommended": d["recommended"],
            "loo_nll": d["ranking"][0]["loo_nll"] if d["ranking"] else None,
            "n_samples": int(X.shape[0]),
            "n_genes": d["n_genes"],
            "n_edges": len(d["edges"]),
            "edges": d["edges"],
            "ranking": d["ranking"][:5],
        })

    report(0.96, "Comparing datasets")
    edge_sets = [{_edge_key(e) for e in p["edges"]} for p in per]

    pairs = []
    for i in range(len(per)):
        for j in range(i + 1, len(per)):
            inter = edge_sets[i] & edge_sets[j]
            union = edge_sets[i] | edge_sets[j]
            sign_i = {_edge_key(e): (e["covariance"] >= 0) for e in per[i]["edges"]}
            sign_j = {_edge_key(e): (e["covariance"] >= 0) for e in per[j]["edges"]}
            agree = sum(1 for k in inter if sign_i[k] == sign_j[k])
            pairs.append({
                "a": per[i]["accession"], "b": per[j]["accession"],
                "shared_genes": len(top_genes),
                "shared_edges": len(inter),
                "edge_jaccard": (len(inter) / len(union)) if union else 0.0,
                "sign_agreement": (agree / len(inter)) if inter else None,
                "same_recommendation": per[i]["recommended"] == per[j]["recommended"],
            })

    # Edges recovered by more than one dataset are the reproducible ones.
    counts: Dict[Tuple[str, str], int] = {}
    for es in edge_sets:
        for k in es:
            counts[k] = counts.get(k, 0) + 1
    recurrent = sorted(
        ({"gene_a": k[0], "gene_b": k[1], "n_datasets": v} for k, v in counts.items() if v > 1),
        key=lambda r: -r["n_datasets"],
    )

    return {
        "source": {"kind": "compare", "accessions": [p["accession"] for p in per]},
        "datasets": per,
        "gene_panel": list(top_genes),
        "comparison": {
            "n_datasets": len(per),
            "n_shared_genes": n_shared,
            "gene_panel_size": len(top_genes),
            "recommendations": {p["accession"]: p["recommended"] for p in per},
            "consensus_recommendation": (
                per[0]["recommended"]
                if len({p["recommended"] for p in per}) == 1 else None
            ),
            "pairs": pairs,
            "recurrent_edges": recurrent[:50],
            "n_recurrent_edges": len(recurrent),
        },
    }


def _band(report: ProgressFn, lo: float, hi: float) -> ProgressFn:
    def scaled(fraction: float, phase: str) -> None:
        report(lo + (hi - lo) * fraction, phase)
    return scaled
