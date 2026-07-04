#!/usr/bin/env python3
"""Parity tests for the ADGENCOV pybind11 bindings.

Every bound C++ function is checked against the ORIGINAL prototype
(ad_covariance_app.py) — and, for the data-driven shrinkers, scikit-learn — to
~1e-9.  This is the Python-side twin of the C++ golden tests: it guarantees the
compiled `_core` module the GUIs and FastAPI service import behaves bit-for-bit
like the reference implementation the Applications Note describes.

Run with the golden venv (which has numpy/sklearn and the built module on the
path):

    PYTHONPATH=python .goldenv/bin/python -m pytest tests/test_bindings.py -q

The build must have produced python/adgencov/_core*.so first
(cmake -B build-py -DADGENCOV_BUILD_PYTHON=ON && cmake --build build-py).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Make the built package importable without installation.
sys.path.insert(0, os.path.join(REPO, "python"))

# Import the original prototype exactly as the C++ golden generators do.
PROTO = os.path.join(
    REPO, "..", "uploads", "ad_extracted",
    "ad_covariance_application_note_v3", "ad_covariance_app.py",
)
_spec = importlib.util.spec_from_file_location("proto", os.path.abspath(PROTO))
proto = importlib.util.module_from_spec(_spec)
sys.modules["proto"] = proto
_spec.loader.exec_module(proto)

try:
    import adgencov
    from adgencov import _core
except Exception as exc:  # pragma: no cover - surfaced as a clear skip reason
    pytest.skip(f"adgencov module not built: {exc}", allow_module_level=True)

TOL = 1e-9


# ---------------------------------------------------------------------------
# Fixtures — the same 6x5 two-block layout used by the C++ golden tests.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20240704)
    n, p = 24, 6
    X = rng.standard_normal((n, p))
    # Induce block structure so AD projection and the recommender do real work.
    X[:, 1] += 0.9 * X[:, 0]
    X[:, 4] += 0.8 * X[:, 3]
    X[:, 5] += 0.5 * X[:, 3]
    labels_str = ["a", "a", "b", "c", "c", "c"]
    labels_int = list(adgencov.factorize(labels_str))
    return dict(X=X, p=p, n=n, labels_str=labels_str, labels_int=labels_int)


def assert_close(a, b, tol=TOL, msg=""):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape} {msg}"
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    assert err <= tol, f"max abs err {err:.3e} > {tol:.0e} {msg}"


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------
def test_reynolds_project(data):
    S = proto.sample_covariance(data["X"])
    got = _core.reynolds_project(S, data["labels_int"])
    want = proto.ad_project_covariance(S, data["labels_str"])
    assert_close(got, want, msg="reynolds_project")


def test_reynolds_symmetric_idempotent(data):
    S = proto.sample_covariance(data["X"])
    P = _core.reynolds_project(S, data["labels_int"])
    assert_close(P, P.T, msg="symmetry")
    assert_close(_core.reynolds_project(P, data["labels_int"]), P, msg="idempotence")


# ---------------------------------------------------------------------------
# estimator family (shrink.hpp)
# ---------------------------------------------------------------------------
def test_sample_covariance(data):
    assert_close(_core.sample_covariance(data["X"]),
                 proto.sample_covariance(data["X"]), msg="sample_covariance")


def test_make_pd(data):
    S = proto.sample_covariance(data["X"])
    assert_close(_core.make_pd(S), proto.make_pd(S), msg="make_pd")


@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.7])
def test_ridge(data, alpha):
    S = proto.sample_covariance(data["X"])
    assert_close(_core.ridge(S, alpha), proto.ridge(S, alpha), msg=f"ridge {alpha}")


@pytest.mark.parametrize("lam,l1", [(0.05, 1.0), (0.1, 0.25), (0.3, 0.5)])
def test_soft_threshold(data, lam, l1):
    S = proto.sample_covariance(data["X"])
    assert_close(_core.soft_threshold_offdiag(S, lam, l1),
                 proto.soft_threshold_offdiag(S, lam, l1),
                 msg=f"soft_threshold {lam},{l1}")


def test_ledoit_wolf(data):
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(data["X"])
    assert abs(_core.ledoit_wolf_shrinkage(data["X"]) - lw.shrinkage_) <= TOL
    assert_close(_core.ledoit_wolf(data["X"]), lw.covariance_, msg="ledoit_wolf cov")


def test_oas(data):
    from sklearn.covariance import OAS
    o = OAS().fit(data["X"])
    assert abs(_core.oas_shrinkage(data["X"]) - o.shrinkage_) <= TOL
    assert_close(_core.oas(data["X"]), o.covariance_, msg="oas cov")


# ---------------------------------------------------------------------------
# model selection (select.hpp) — the AD variants especially
# ---------------------------------------------------------------------------
ALL_METHODS = [
    ("sample", {}), ("ad_sample", {}),
    ("ridge", {"alpha": 0.3}), ("ad_ridge", {"alpha": 0.3}),
    ("lasso", {"lam": 0.05}), ("ad_lasso", {"lam": 0.05}),
    ("elastic_net", {"lam": 0.05, "l1_ratio": 0.25}),
    ("ad_elastic_net", {"lam": 0.05, "l1_ratio": 0.25}),
    ("lw", {}), ("ledoit_wolf", {}), ("oas", {}),
    ("ad_linear_lw", {}), ("ad_oas", {}),
]


@pytest.mark.parametrize("method,params", ALL_METHODS)
def test_estimate_covariance(data, method, params):
    got = _core.estimate_covariance(data["X"], data["labels_int"], method, params)
    want = proto.estimate_covariance(data["X"], data["labels_str"], method, params)
    assert_close(got, want, msg=f"estimate_covariance {method}")


def test_gaussian_nll_one(data):
    X = data["X"]
    mu = X.mean(axis=0)
    Sigma = proto.estimate_covariance(X, data["labels_str"], "ad_ridge", {"alpha": 0.3})
    got = _core.gaussian_nll_one(X[0], mu, Sigma)
    want = proto.gaussian_nll_one(X[0], mu, Sigma)
    assert abs(got - want) <= TOL, f"nll {got} vs {want}"


@pytest.mark.parametrize("method,params", ALL_METHODS)
def test_loo_nll(data, method, params):
    got = _core.loo_nll(data["X"], data["labels_int"], method, params)
    want = proto.loo_nll(data["X"], data["labels_str"], method, params)
    assert np.isfinite(got) == np.isfinite(want)
    if np.isfinite(want):
        assert abs(got - want) <= 1e-8, f"loo_nll {method}: {got} vs {want}"


def test_candidate_grid(data):
    grid = _core.candidate_grid(data["p"], data["n"])
    ref = proto.candidate_grid(data["p"], data["n"])
    assert len(grid) == len(ref)
    for spec, (method, params) in zip(grid, ref):
        assert spec.method == method
        assert dict(spec.params) == pytest.approx(params)


def test_recommend_estimator_ranking(data):
    got = _core.recommend_estimator(data["X"], data["labels_int"])
    want = proto.recommend_estimator(data["X"], data["labels_str"])
    # Same set of surviving candidates and same order (ascending LOO-NLL).
    assert [r.spec.method for r in got] == [r.name for r in want]
    for gr, wr in zip(got, want):
        assert abs(gr.loo_nll - wr.loo_nll) <= 1e-8, f"{gr.spec.method} loo"
        assert_close(gr.covariance, wr.covariance, msg=f"{gr.spec.method} cov")


# ---------------------------------------------------------------------------
# clustering (clustering.hpp) vs scikit-learn
# ---------------------------------------------------------------------------
def _relabel_canonical(labels):
    """Map ids to first-appearance order so partitions compare id-invariantly."""
    remap, out = {}, []
    for v in labels:
        if v not in remap:
            remap[v] = len(remap)
        out.append(remap[v])
    return out


@pytest.mark.parametrize("k", [2, 3, 4])
def test_agglomerative_average(data, k):
    from sklearn.cluster import AgglomerativeClustering
    X = data["X"]
    corr = np.corrcoef(X, rowvar=False)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    got = _core.agglomerative_average(dist, k)
    want = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(dist)
    assert _relabel_canonical(got) == _relabel_canonical(want), f"k={k}"


# ---------------------------------------------------------------------------
# high-level API glue
# ---------------------------------------------------------------------------
def test_gene_family_label():
    for gene in ["HSPA1A", "COL1A1", "MT-CO1", "ABC123", "xyz"]:
        assert _core.gene_family_label(gene) == proto.gene_family_label(gene)


def test_top_edges_matches_prototype(data):
    Sigma = proto.estimate_covariance(data["X"], data["labels_str"], "oas", {})
    genes = [f"g{i}" for i in range(data["p"])]
    got = adgencov.top_edges(Sigma, genes, top_fraction=0.2)
    want = proto.top_edges(Sigma, genes, top_fraction=0.2)
    assert len(got) == len(want)
    for e, (_, row) in zip(got, want.iterrows()):
        assert e.gene_a == row.gene_a and e.gene_b == row.gene_b
        assert abs(e.covariance - row.covariance) <= TOL


def test_analyze_end_to_end(data):
    genes = [f"g{i}" for i in range(data["p"])]
    result = adgencov.analyze(data["X"], data["labels_int"], genes=genes)
    ref = proto.recommend_estimator(data["X"], data["labels_str"])
    assert result.best.spec.method == ref[0].name
    d = result.to_dict()
    assert d["recommended"] == ref[0].name
    assert len(d["ranking"]) == len(ref)
