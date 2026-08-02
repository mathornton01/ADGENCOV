#!/usr/bin/env python3
"""Synthetic probe: when does the AD symmetry prior help, and when does it not?

Generates data under four regimes and compares the best AD estimator against the
best ordinary shrinkage estimator by held-out Gaussian negative log-likelihood.
Both estimators are selected on the training split by exact leave-one-out NLL,
so the comparison is between two honest model-selection procedures rather than
between two fixed estimators.

Regimes:
  matched block      true covariance is block-exchangeable on the SAME partition
                     handed to the estimator;
  weak block         same, but with the block contrast scaled down;
  mismatched block   true covariance is block-exchangeable on a DIFFERENT
                     partition than the one handed to the estimator;
  dense unstructured no block structure at all (random dense SPD).

Usage:
    PYTHONPATH=python python3 scripts/synthetic_probe.py [--reps 8] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import adgencov
from adgencov._core import gaussian_nll_one

P, N_TRAIN, N_TEST = 40, 20, 200
AD_MODES = ["none", "projection", "target", "optimal"]


def block_cov(labels, within, cross, rng, jitter=0.05):
    """Block-exchangeable covariance: `within` inside a block, `cross` between."""
    p = len(labels)
    S = np.full((p, p), cross, dtype=float)
    for i in range(p):
        for j in range(p):
            if labels[i] == labels[j]:
                S[i, j] = within
    S += jitter * rng.standard_normal((p, p))
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return spd(S)


def spd(S, floor=0.05):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    return V @ np.diag(np.maximum(w, floor)) @ V.T


def dense_cov(p, rng):
    A = rng.standard_normal((p, p))
    return spd(A @ A.T / p)


def regimes(rng):
    labels = [i // (P // 4) for i in range(P)]
    other = [i % 4 for i in range(P)]           # a genuinely different partition
    return {
        "Matched block":     (labels, block_cov(labels, 0.9, 0.1, rng)),
        "Weak block":        (labels, block_cov(labels, 0.35, 0.15, rng)),
        "Mismatched block":  (labels, block_cov(other, 0.9, 0.1, rng)),
        "Dense unstructured": (labels, dense_cov(P, rng)),
    }


def held_out_nll(Sigma, mu, Xte):
    return float(np.mean([gaussian_nll_one(x, mu, Sigma) for x in Xte]))


def best_of(Xtr, Xte, labels, modes):
    """Select on the training split by LOO, then score on the held-out split."""
    res = adgencov.analyze(Xtr, labels, criterion="loo", ad_modes=modes, sweep=True)
    d = res.to_dict()
    best = d["ranking"][0]
    Sigma = np.asarray(
        adgencov.estimate_covariance(Xtr, labels, best["method"], best["params"]),
        dtype=float)
    return best, held_out_nll(Sigma, Xtr.mean(axis=0), Xte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", default="data/synthetic_probe")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    for name in ("Matched block", "Weak block", "Mismatched block", "Dense unstructured"):
        wins, deltas, picks = 0, [], []
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 + rep)
            labels, Sigma = regimes(rng)[name]
            L = np.linalg.cholesky(Sigma)
            Xtr = rng.standard_normal((N_TRAIN, P)) @ L.T
            Xte = rng.standard_normal((N_TEST, P)) @ L.T
            ad, ad_nll = best_of(Xtr, Xte, labels, ["projection", "target", "optimal"])
            od, od_nll = best_of(Xtr, Xte, labels, ["none"])
            deltas.append(ad_nll - od_nll)
            picks.append(ad["method"])
            if ad_nll < od_nll:
                wins += 1
        mean_d = float(np.mean(deltas))
        summary[name] = {"wins": wins, "reps": args.reps, "mean_delta": mean_d,
                         "deltas": [float(x) for x in deltas]}
        print(f"{name:<20} AD wins {wins}/{args.reps}   mean AD-ordinary NLL = {mean_d:+.2f}")

    tex = ["\\begin{tabular}{lcr}", "\\toprule",
           "Scenario & AD wins & Mean AD $-$ ordinary NLL\\\\", "\\midrule"]
    for name, s in summary.items():
        tex.append(f"{name} & {s['wins']}/{s['reps']} & {s['mean_delta']:+.2f}\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(tex)
    print("\n--- Supplementary Table S1 (LaTeX) ---\n" + table)

    with open(os.path.join(args.out, "synthetic_probe.tex"), "w") as fh:
        fh.write(table + "\n")
    with open(os.path.join(args.out, "synthetic_probe.json"), "w") as fh:
        json.dump({"p": P, "n_train": N_TRAIN, "n_test": N_TEST,
                   "reps": args.reps, "regimes": summary}, fh, indent=1)
    print(f"\nwrote {args.out}/synthetic_probe.tex and synthetic_probe.json")


if __name__ == "__main__":
    main()
