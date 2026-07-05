#!/usr/bin/env python3
"""Generate golden reference values for the ADGENCOV recommender/likelihood tests.

Imports the ORIGINAL prototype (ad_covariance_app.py) so the C++ port of the
selection layer — estimate_covariance, gaussian_nll_one, loo_nll, candidate_grid
and recommend_estimator — is validated against the exact numerics the
Applications Note describes.

Outputs tests/golden_select.hpp.  Regenerate with:
    .goldenv/bin/python tests/gen_golden_select.py
"""
from __future__ import annotations
import importlib.util
import os
import sys

import numpy as np

import adtarget_ref  # shared AD-target reference (same dir on sys.path)

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(
    HERE, "..", "..", "uploads", "ad_extracted",
    "ad_covariance_application_note_v3", "ad_covariance_app.py",
)

spec = importlib.util.spec_from_file_location("proto", os.path.abspath(PROTO))
proto = importlib.util.module_from_spec(spec)
sys.modules["proto"] = proto
spec.loader.exec_module(proto)

# Same fixed fixture as gen_golden.py: n=6 samples, p=5 genes, two blocks.
X = np.array([
    [ 1.20, -0.30,  2.10,  0.05, -1.10],
    [ 0.40,  0.90, -0.60,  1.30,  0.70],
    [-1.50,  0.20,  0.80, -0.40,  1.60],
    [ 2.30, -1.10,  1.40,  0.90, -0.50],
    [-0.70,  1.50, -1.20, -1.00,  2.20],
    [ 0.60,  0.10,  0.30, -0.20, -0.90],
], dtype=float)
n, p = X.shape
labels = [0, 0, 1, 1, 1]

# ---------------------------------------------------------------------------
# Signature: stable, matches the C++ side's formatting of an EstimatorSpec.
# ---------------------------------------------------------------------------
def sig(name, params):
    if not params:
        return name
    items = ",".join(f"{k}={params[k]:g}" for k in sorted(params))
    return f"{name}[{items}]"


# ---------------------------------------------------------------------------
# gaussian_nll_one on a concrete (x, mu, Sigma).
# ---------------------------------------------------------------------------
mu_full = X.mean(axis=0)
Sigma_case = proto.estimate_covariance(X, labels, "ad_ridge", {"alpha": 0.2})
nll_case = proto.gaussian_nll_one(X[0], mu_full, Sigma_case)

# ---------------------------------------------------------------------------
# Rank-1 downdate targets: np.cov(train, ddof=1) for a couple of held-out rows.
# ---------------------------------------------------------------------------
def loo_cov(i):
    return proto.sample_covariance(np.delete(X, i, axis=0))

def loo_mu(i):
    return np.delete(X, i, axis=0).mean(axis=0)

# ---------------------------------------------------------------------------
# loo_nll per representative method.
# ---------------------------------------------------------------------------
loo_specs = [
    ("ad_ridge",       {"alpha": 0.2}),
    ("ad_lasso",       {"lam": 0.1}),
    ("ad_elastic_net", {"lam": 0.1, "l1_ratio": 0.25}),
    ("lw",             {}),
    ("oas",            {}),
    ("ad_linear_lw",   {}),
    ("ad_oas",         {}),
    ("ad_target_lw",   {}),
    ("ad_target_oas",  {}),
    ("ad_target_ridge", {"lam": 0.5}),
]


def _loo(m, pr):
    if adtarget_ref.is_ad_target(m):
        return adtarget_ref.loo_nll(proto, X, labels, m, pr)
    return proto.loo_nll(X, labels, m, pr)


loo_vals = [(sig(m, pr), _loo(m, pr)) for m, pr in loo_specs]

# ---------------------------------------------------------------------------
# estimate_covariance dispatcher outputs (make_pd'd) for a few methods.
# ---------------------------------------------------------------------------
est_specs = [
    ("est_ad_ridge_0p2",  "ad_ridge",       {"alpha": 0.2}),
    ("est_ad_lasso_0p1",  "ad_lasso",       {"lam": 0.1}),
    ("est_lw",            "lw",             {}),
    ("est_oas",           "oas",            {}),
    ("est_ad_linear_lw",  "ad_linear_lw",   {}),
    ("est_ad_oas",        "ad_oas",         {}),
    ("est_ad_target_ridge_0p5", "ad_target_ridge", {"lam": 0.5}),
    ("est_ad_target_lw",  "ad_target_lw",   {}),
    ("est_ad_target_oas", "ad_target_oas",  {}),
]


def _est(m, pr):
    if adtarget_ref.is_ad_target(m):
        return adtarget_ref.estimate(proto, X, labels, m, pr)
    return proto.estimate_covariance(X, labels, m, pr)


est_mats = [(tag, _est(m, pr)) for tag, m, pr in est_specs]

# ---------------------------------------------------------------------------
# Full recommend_estimator ranking over the EXTENDED grid (prototype grid + the
# AD-target family), sorted by loo_nll ascending — mirrors candidate_grid().
# ---------------------------------------------------------------------------
rec = adtarget_ref.ranked(proto, X, labels)
rec_rows = [(row[0], row[3], row[4]) for row in rec]  # (sig, loo_nll, cond)

# ---------------------------------------------------------------------------
# Emit C++ header.
# ---------------------------------------------------------------------------
def fmt_mat(name, M):
    M = np.asarray(M, dtype=float)
    r, c = M.shape
    flat = ", ".join(f"{v:.17g}" for v in M.flatten(order="C"))
    return (f"inline const GoldenMat {name} {{\n"
            f"    {r}, {c},\n"
            f"    {{{flat}}}\n}};\n")

L = []
L.append("// AUTO-GENERATED by tests/gen_golden_select.py — do not edit by hand.")
L.append("// Reference values from the prototype ad_covariance_app.py.")
L.append("#ifndef ADGENCOV_GOLDEN_SELECT_HPP")
L.append("#define ADGENCOV_GOLDEN_SELECT_HPP")
L.append("#include <string>")
L.append("#include <vector>")
L.append("")
L.append("namespace golden_select {")
L.append("")
L.append("struct GoldenMat { int rows; int cols; std::vector<double> data; };")
L.append("")
L.append(f"inline const int kN = {n};")
L.append(f"inline const int kP = {p};")
Xflat = ", ".join(f"{v:.17g}" for v in X.flatten(order="C"))
L.append(f"inline const std::vector<double> kX {{{Xflat}}};  // {n}x{p} row-major")
L.append(f"inline const std::vector<int> kLabels {{{', '.join(str(x) for x in labels)}}};")
L.append("")
# gaussian_nll_one case
Sflat = ", ".join(f"{v:.17g}" for v in np.asarray(Sigma_case).flatten(order='C'))
L.append(f"inline const std::vector<double> kNllSigma {{{Sflat}}};  // {p}x{p} row-major")
muflat = ", ".join(f"{v:.17g}" for v in mu_full)
L.append(f"inline const std::vector<double> kNllMu {{{muflat}}};")
x0flat = ", ".join(f"{v:.17g}" for v in X[0])
L.append(f"inline const std::vector<double> kNllX {{{x0flat}}};")
L.append(f"inline const double kNllValue = {nll_case:.17g};")
L.append("")
# downdate targets
L.append(fmt_mat("loo_cov_drop0", loo_cov(0)))
L.append(fmt_mat("loo_cov_drop3", loo_cov(3)))
m0 = ", ".join(f"{v:.17g}" for v in loo_mu(0))
L.append(f"inline const std::vector<double> loo_mu_drop0 {{{m0}}};")
L.append("")
# loo_nll per method
L.append(f"inline const std::vector<std::string> kLooSigs {{{', '.join(chr(34)+s+chr(34) for s,_ in loo_vals)}}};")
L.append(f"inline const std::vector<double> kLooVals {{{', '.join(f'{v:.17g}' for _,v in loo_vals)}}};")
L.append("")
# dispatcher matrices
for tag, M in est_mats:
    L.append(fmt_mat(tag, M))
# recommend ranking
L.append(f"inline const std::vector<std::string> kRecSigs {{{', '.join(chr(34)+s+chr(34) for s,_,_ in rec_rows)}}};")
L.append(f"inline const std::vector<double> kRecLoo {{{', '.join(f'{v:.17g}' for _,v,_ in rec_rows)}}};")
L.append(f"inline const std::vector<double> kRecCond {{{', '.join(f'{c:.17g}' for _,_,c in rec_rows)}}};")
L.append("")
L.append("}  // namespace golden_select")
L.append("")
L.append("#endif  // ADGENCOV_GOLDEN_SELECT_HPP")

out = os.path.join(HERE, "golden_select.hpp")
with open(out, "w") as f:
    f.write("\n".join(L))
print(f"Wrote {out}")
print(f"  gaussian_nll_one case value = {nll_case:.6f}")
print(f"  {len(rec_rows)} ranked candidates; best = {rec_rows[0][0]} @ {rec_rows[0][1]:.4f}")
