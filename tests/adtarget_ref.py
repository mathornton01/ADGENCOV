"""Independent Python reference for the symmetry-target ("AD-target") estimators.

The frozen prototype (ad_covariance_app.py) predates these estimators, so this
module supplies the reference numerics for them, built entirely from the
prototype's own validated primitives (sample_covariance, ad_project_covariance,
ridge, make_pd, gaussian_nll_one) plus scikit-learn's LW/OAS intensities.  It is
shared by both the C++ golden generator (gen_golden_select.py) and the pybind11
parity suite (test_bindings.py) so the two validate the C++ against the same
reference the user's uploaded ad_target_estimators.py describes.

Conventions match the C++ core exactly:
  * S is the unbiased (ddof=1) sample covariance, as everywhere in ADGENCOV;
  * ad_target_ridge  : (1-lam) S + lam P_G(S), then identity ridge diag_alpha
                       (default 1e-3);
  * ad_target_lw/oas : lam = LW/OAS(X); (1-lam) S + lam P_G(S); diag_alpha 0.
"""
from __future__ import annotations

import numpy as np

# The seven candidates appended to candidate_grid, in C++ order.
AD_TARGET_SPECS = [
    ("ad_target_lw", {}),
    ("ad_target_oas", {}),
    ("ad_target_ridge", {"lam": 0.1}),
    ("ad_target_ridge", {"lam": 0.3}),
    ("ad_target_ridge", {"lam": 0.5}),
    ("ad_target_ridge", {"lam": 0.7}),
    ("ad_target_ridge", {"lam": 0.9}),
]


def is_ad_target(method: str) -> bool:
    return method.startswith("ad_target_")


def _lam(proto, X, method, params):
    from sklearn.covariance import LedoitWolf, OAS

    if method == "ad_target_ridge":
        return float(params.get("lam", 0.5))
    if method == "ad_target_lw":
        return float(LedoitWolf().fit(np.asarray(X, dtype=float)).shrinkage_)
    if method == "ad_target_oas":
        return float(OAS().fit(np.asarray(X, dtype=float)).shrinkage_)
    raise ValueError(f"not an ad_target method: {method}")


def estimate(proto, X, labels, method, params):
    """Reference estimate_covariance for an ad_target method (make_pd'd)."""
    X = np.asarray(X, dtype=float)
    S = proto.sample_covariance(X)                    # ddof=1, == C++ unbiased S
    PG = proto.ad_project_covariance(S, labels)       # Reynolds projection
    lam = min(1.0, max(0.0, _lam(proto, X, method, params)))
    C = (1.0 - lam) * S + lam * PG
    diag_alpha = float(params.get("diag_alpha",
                                  1e-3 if method == "ad_target_ridge" else 0.0))
    if diag_alpha > 0.0:
        C = proto.ridge(C, diag_alpha)                # identity shrink, == C++ ridge
    return proto.make_pd(C)


def loo_nll(proto, X, labels, method, params):
    """Reference leave-one-out mean Gaussian NLL for an ad_target method.

    Brute-force (rebuild each training submatrix); the C++ fast path uses a
    rank-1 downdate that agrees to ~1e-11, well inside the 1e-9 test tolerance.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    total = 0.0
    for i in range(n):
        Xtr = np.delete(X, i, axis=0)
        mu = Xtr.mean(axis=0)
        C = estimate(proto, Xtr, labels, method, params)
        total += proto.gaussian_nll_one(X[i], mu, C)
    return float(total / n)


def extended_grid(proto, p, n):
    """The full C++ candidate_grid: prototype grid + AD_TARGET_SPECS, in order."""
    base = [(m, dict(pr)) for m, pr in proto.candidate_grid(p, n)]
    return base + [(m, dict(pr)) for m, pr in AD_TARGET_SPECS]


def _sig(name, params):
    if not params:
        return name
    items = ",".join(f"{k}={params[k]:g}" for k in sorted(params))
    return f"{name}[{items}]"


def ranked(proto, X, labels):
    """Reproduce recommend_estimator over the extended grid.

    Uses the prototype for its own methods and this module for ad_target, then
    ranks ascending by LOO-NLL with a stable sort (matching the C++ path).
    Returns a list of (sig, method, params, loo_nll, condition_number).
    """
    X = np.asarray(X, dtype=float)
    p, n = X.shape[1], X.shape[0]
    rows = []
    for method, params in extended_grid(proto, p, n):
        try:
            if is_ad_target(method):
                loo = loo_nll(proto, X, labels, method, params)
                Sigma = estimate(proto, X, labels, method, params)
            else:
                loo = float(proto.loo_nll(X, labels, method, params))
                Sigma = np.asarray(
                    proto.estimate_covariance(X, labels, method, params), dtype=float)
            ev = np.abs(np.linalg.eigvalsh(np.asarray(Sigma, dtype=float)))
            lo = float(ev.min())
            cond = (float(ev.max()) / lo) if lo > 0.0 else float("inf")
            rows.append([_sig(method, params), method, dict(params), loo, cond])
        except Exception:  # noqa: BLE001 - skip candidates that fail to estimate
            continue
    # Stable ascending sort by LOO-NLL, matching std::stable_sort.
    order = sorted(range(len(rows)), key=lambda k: rows[k][3])
    return [rows[k] for k in order]
