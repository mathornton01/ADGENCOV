#!/usr/bin/env python3
"""Reproduce the GSE52778 example reported in the ADGENCOV application note.

Runs the documented protocol end-to-end through the compiled core and emits the
paper's Table 1 (as LaTeX) plus the Figure 1 network data, so the manuscript can
be regenerated from real output rather than transcribed by hand.

Protocol (Sec. "Example: airway smooth muscle RNA-seq"):
  * submitter-supplied FPKM matrix, the 16 individual ``*_LL*`` sample columns
    (the pooled ``*_FPKM`` / ``*_conf_lo`` / ``*_conf_hi`` Cufflinks columns are
    NOT samples and must be excluded);
  * duplicate gene symbols collapsed by highest mean expression;
  * log2(FPKM + 1); top 64 variable genes; per-gene z-score;
  * correlation-derived 4-block surrogate symmetry;
  * estimators ranked by exact leave-one-out Gaussian NLL.

Usage:
    PYTHONPATH=python python3 scripts/paper_gse52778.py [--outdir paper_out]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import urllib.request

import numpy as np

import adgencov
from adgencov._core import build_group_labels, factorize, load_expression_matrix, preprocess

FPKM_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52778/suppl/"
    "GSE52778_All_Sample_FPKM_Matrix.txt.gz"
)
SAMPLE_REGEX = "_LL[0-9]+"      # the 16 individual samples
GENE_COL = "gene_short_name"
N_GENES = 64
N_BLOCKS = 4
MIN_MEAN = 0.1
TOP_FRACTION = 0.01

# Paper Table 1 -> code method names.  Equation (2) of the note defines the AD
# family as the convex combination Sigma_AD = (1-lam) S + lam P_G(S) followed by
# ridge/LW/OAS: that is exactly the ``ad_target_*`` family.  The ``ad_*`` family
# is the lam=1 hard-projection special case, reported separately for contrast.
TABLE_ROWS = [
    ("AD-Ridge (Eq. 2)", "ad_target_ridge", None),
    ("AD-Ledoit--Wolf (Eq. 2)", "ad_target_lw", None),
    ("AD-OAS (Eq. 2)", "ad_target_oas", None),
    ("ordinary OAS", "oas", None),
    ("ordinary Ledoit--Wolf", "lw", None),
    ("AD-Ridge, hard projection ($\\lambda=1$)", "ad_ridge", 0.4),
]


def fetch_matrix(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    gz = os.path.join(cache_dir, "GSE52778_All_Sample_FPKM_Matrix.txt.gz")
    txt = gz[:-3]
    if not os.path.exists(txt):
        if not os.path.exists(gz):
            print(f"downloading {FPKM_URL} ...")
            urllib.request.urlretrieve(FPKM_URL, gz)
        with gzip.open(gz, "rb") as fi, open(txt, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    path = fetch_matrix(args.outdir)
    data = load_expression_matrix(path, sample_regex=SAMPLE_REGEX, gene_col=GENE_COL)
    ds = preprocess(data, n_genes=N_GENES, min_mean=MIN_MEAN, log_transform=True)
    X = np.asarray(ds.X, dtype=float)
    print(f"matrix: {X.shape[0]} samples x {X.shape[1]} genes")

    names = build_group_labels(ds, "correlation_blocks", n_blocks=N_BLOCKS)
    codes = factorize(names)
    sizes = sorted(np.bincount(np.asarray(codes)).tolist())
    print("correlation-derived block sizes:", sizes)

    res = adgencov.analyze(
        X, codes, genes=list(ds.genes), top_fraction=TOP_FRACTION, criterion="loo"
    )
    d = res.to_dict()

    rank_of = {}
    for i, r in enumerate(d["ranking"]):
        key = (r["method"], round(float(r["params"].get("alpha", r["params"].get("lam", -1))), 6))
        rank_of.setdefault(r["method"], (i + 1, r))
        rank_of[key] = (i + 1, r)

    print(f"\nrecommended: {d['recommended']}   (n edges = {len(d['edges'])})")
    print("\nfull ranking (top 12):")
    for i, r in enumerate(d["ranking"][:12]):
        p = ",".join(f"{k}={v:g}" for k, v in sorted(r["params"].items()))
        print(f"  {i+1:2d}. {r['method']:<18}{p:<24}{r['loo_nll']:.3f}")

    # ---- Table 1 (LaTeX) ---------------------------------------------------
    rows = []
    for label, method, param in TABLE_ROWS:
        best = None
        for i, r in enumerate(d["ranking"]):
            if r["method"] != method:
                continue
            if param is not None:
                v = r["params"].get("alpha", r["params"].get("lam"))
                if v is None or abs(float(v) - param) > 1e-9:
                    continue
            if best is None or r["loo_nll"] < best[1]["loo_nll"]:
                best = (i + 1, r)
        if best is None:
            continue
        rank, r = best
        lam = r["params"].get("lam")
        show = label
        if method == "ad_target_ridge" and lam is not None:
            show = f"AD-Ridge (Eq. 2, $\\lambda={lam:g}$)"
        rows.append((rank, show, r["loo_nll"]))
    rows.sort(key=lambda t: t[2])

    tex = ["\\begin{tabular}{lll}", "\\toprule",
           "Rank & Method & Mean LOO-NLL\\\\", "\\midrule"]
    for i, (_, label, nll) in enumerate(rows, start=1):
        tex.append(f"{i} & {label} & {nll:.3f}\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(tex)
    print("\n--- Table 1 (LaTeX) ---\n" + table)

    with open(os.path.join(args.outdir, "table1.tex"), "w") as fh:
        fh.write(table + "\n")
    with open(os.path.join(args.outdir, "gse52778_result.json"), "w") as fh:
        json.dump({"block_sizes": sizes, "n_samples": int(X.shape[0]), **d}, fh)
    print(f"\nwrote {args.outdir}/table1.tex and {args.outdir}/gse52778_result.json")


if __name__ == "__main__":
    main()
