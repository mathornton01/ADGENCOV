#!/usr/bin/env python3
"""Treatment-stratified AD covariance for GSE52778 (Supplementary Section S5).

GSE52778 contains four airway smooth muscle cell lines under four conditions.
Estimating covariance separately within each condition leaves only four samples
per estimate, so the output is descriptive rather than confirmatory; the point is
to show whether the covariance structure reorganizes across conditions rather
than simply rescaling.

The submitter-supplied FPKM matrix orders the sixteen individual ``*_LL*`` sample
columns in contiguous blocks of four by condition -- Dex, Alb, Untreated,
Alb+Dex -- which this script reads from the header rather than assuming.

Usage:
    PYTHONPATH=python python3 scripts/treatment_rewiring.py [--outdir data/gse52778]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

import adgencov
from adgencov._core import build_group_labels, factorize, load_expression_matrix, preprocess

MATRIX = "GSE52778_All_Sample_FPKM_Matrix.txt"
SAMPLE_REGEX = "_LL[0-9]+"
GENE_COL = "gene_short_name"
N_GENES, MIN_MEAN, TOP = 64, 0.1, 100
METHOD, PARAMS = "ad_target_ridge", {"lam": 0.7}
# Header prefixes of the individual sample columns, in file order.
PREFIXES = [("Dex", "Dex_LL"), ("Alb", "Alb_LL"),
            ("untreated", "Untreated_LL"), ("Alb+Dex", "Alb_Dex_LL")]


def sample_conditions(path: str):
    """Map each retained sample column to its condition, from the file header."""
    with open(path) as fh:
        # Split on any whitespace: the matrix is tab-delimited but no column name
        # contains a space, so this works whichever delimiter the copy carries.
        header = fh.readline().split()
    cols = [c for c in header if "_LL" in c]
    groups: dict[str, list[int]] = {name: [] for name, _ in PREFIXES}
    for i, c in enumerate(cols):
        for name, pref in PREFIXES:
            # Alb_Dex_LL must be tested before Alb_LL; PREFIXES order handles it
            # because Alb_LL never matches an Alb_Dex_LL column.
            if c.startswith(pref):
                groups[name].append(i)
                break
    return cols, groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/gse52778")
    ap.add_argument("--top", type=int, default=TOP)
    args = ap.parse_args()
    path = os.path.join(args.outdir, MATRIX)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run scripts/paper_gse52778.py first.")

    cols, groups = sample_conditions(path)
    print("sample columns:", cols)
    print("condition -> indices:", groups)
    if sorted(i for v in groups.values() for i in v) != list(range(len(cols))):
        raise SystemExit("could not assign every sample column to a condition")

    data = load_expression_matrix(path, sample_regex=SAMPLE_REGEX, gene_col=GENE_COL)
    ds = preprocess(data, n_genes=N_GENES, min_mean=MIN_MEAN, log_transform=True)
    X = np.asarray(ds.X, dtype=float)
    genes = list(ds.genes)
    codes = factorize(build_group_labels(ds, "gene_family", n_blocks=4))

    conds = [name for name, _ in PREFIXES]
    tops, mats = {}, {}
    for cond in conds:
        Xc = X[groups[cond], :]
        Sigma = np.asarray(
            adgencov.estimate_covariance(Xc, codes, METHOD, PARAMS), dtype=float)
        mats[cond] = Sigma
        pairs = sorted(((i, j, Sigma[i, j])
                        for i, j in itertools.combinations(range(len(genes)), 2)),
                       key=lambda t: -abs(t[2]))
        tops[cond] = pairs[:args.top]
        print(f"{cond:<10} n={len(groups[cond])}  |cov| over top-{args.top}: "
              f"{abs(pairs[0][2]):.3f} .. {abs(pairs[args.top-1][2]):.3f}")

    sets = [{(i, j) for i, j, _ in tops[c]} for c in conds]
    union = set().union(*sets)
    in_all = set.intersection(*sets)
    sign_changes = sum(
        1 for (i, j) in union
        if len({np.sign(mats[c][i, j]) for c in conds if mats[c][i, j] != 0}) > 1)

    print(f"\nunion of four top-{args.top} lists : {len(union)} distinct pairs")
    print(f"pairs present in all four     : {len(in_all)}")
    print(f"pairs with a sign change      : {sign_changes}")

    ranked = sorted(union, key=lambda p: -max(abs(mats[c][p[0], p[1]]) for c in conds))
    print("\nmost condition-specific pairs:")
    for (i, j) in ranked[:8]:
        vals = "  ".join(f"{c}={mats[c][i, j]:+.2f}" for c in conds)
        print(f"  {genes[i]:<12}{genes[j]:<12} {vals}")

    out = os.path.join(args.outdir, "treatment_rewiring.json")
    with open(out, "w") as fh:
        json.dump({"top_n": args.top, "conditions": conds,
                   "union": len(union), "in_all": len(in_all),
                   "sign_changes": sign_changes,
                   "top_pairs": [{"a": genes[i], "b": genes[j],
                                  **{c: float(mats[c][i, j]) for c in conds}}
                                 for i, j in ranked[:12]]}, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
