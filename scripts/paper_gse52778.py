#!/usr/bin/env python3
"""Reproduce the GSE52778 example reported in the ADGENCOV application note.

Runs the documented protocol end-to-end through the compiled core and emits the
paper's Table 1 (as LaTeX), the Figure 1 network payload, and the derived
analysis outputs referenced by the manuscript's Data Availability statement, so
the manuscript can be regenerated from real output rather than transcribed by
hand.

Protocol (Sec. "Example: airway smooth muscle RNA-seq"):
  * submitter-supplied FPKM matrix, the 16 individual ``*_LL*`` sample columns
    (the pooled ``*_FPKM`` / ``*_conf_lo`` / ``*_conf_hi`` Cufflinks columns are
    NOT samples and must be excluded);
  * duplicate gene symbols collapsed by highest mean expression;
  * log2(FPKM + 1); top 64 variable genes; per-gene z-score;
  * a gene-family symmetry (``--group``; ``correlation_blocks`` reproduces the
    data-driven surrogate reported in the Supplementary Information);
  * the full estimator grid, including every Algebraic-Diversity mode -- ordinary
    (``none``), hard projection (``ad_*``), Eq. (2) symmetry target
    (``ad_target_*``) and the Eq. (3) optimal weight (``ad_target_optimal``) --
    ranked by exact leave-one-out Gaussian NLL.

Usage:
    PYTHONPATH=python python3 scripts/paper_gse52778.py [--outdir paper_out]
                                                        [--group gene_family]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import urllib.request
from collections import Counter

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
TOP_FRACTION = 0.01   # overridable via --top-fraction
AD_MODES = ["none", "projection", "target", "optimal"]

# Paper Table 1 -> code method names.  Equation (2) of the note defines the AD
# family as the convex combination Sigma_AD = (1-lam) S + lam P_G(S) followed by
# ridge/LW/OAS: that is exactly the ``ad_target_*`` family.  The ``ad_*`` family
# is the lam=1 hard-projection special case, reported separately for contrast.
# ``ad_target_optimal`` is the Eq. (3) derived weight (arXiv:2605.17111 Prop 3.2).
TABLE_ROWS = [
    ("AD-Ridge (Eq. 2)", "ad_target_ridge", None),
    ("AD-Ledoit--Wolf (Eq. 2)", "ad_target_lw", None),
    ("AD-OAS (Eq. 2)", "ad_target_oas", None),
    ("AD-optimal $\\alpha^\\ast$ (Eq. 3)", "ad_target_optimal", None),
    ("ordinary ridge", "ridge", None),
    ("ordinary OAS", "oas", None),
    ("ordinary Ledoit--Wolf", "lw", None),
    ("AD-Ridge, hard projection ($\\lambda=1$)", "ad_ridge", None),
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


def write_csv(path: str, header, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {os.path.basename(path)}")


def louvain_communities(edges, genes):
    """Louvain communities over the retained edge set (seed=0, as in figure.py)."""
    try:
        import networkx as nx
    except ImportError:
        return None
    G = nx.Graph()
    G.add_nodes_from(genes)
    for e in edges:
        G.add_edge(e["gene_a"], e["gene_b"], w=abs(float(e["covariance"])))
    comms = sorted(nx.community.louvain_communities(G, weight="w", seed=0),
                   key=len, reverse=True)
    return {n: i for i, c in enumerate(comms) for n in c}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper_out")
    ap.add_argument("--group", default="gene_family",
                    help="Symmetry partition (gene_family, correlation_blocks, none, ...).")
    ap.add_argument("--n-blocks", type=int, default=N_BLOCKS,
                    help="Block count for the clustering-based partitions.")
    ap.add_argument("--n-genes", type=int, default=N_GENES)
    ap.add_argument("--top-fraction", type=float, default=TOP_FRACTION,
                    help="Fraction of gene pairs kept as network edges.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    path = fetch_matrix(args.outdir)
    data = load_expression_matrix(path, sample_regex=SAMPLE_REGEX, gene_col=GENE_COL)
    ds = preprocess(data, n_genes=args.n_genes, min_mean=MIN_MEAN, log_transform=True)
    X = np.asarray(ds.X, dtype=float)
    print(f"matrix: {X.shape[0]} samples x {X.shape[1]} genes")

    if args.group == "none":
        names = ["all"] * X.shape[1]
    else:
        names = build_group_labels(ds, args.group, n_blocks=args.n_blocks)
    codes = factorize(names)
    sizes = sorted(np.bincount(np.asarray(codes)).tolist())
    print(f"symmetry '{args.group}' group sizes: {sizes}")
    print(f"  groups: {Counter(names).most_common()}")

    res = adgencov.analyze(
        X, codes, genes=list(ds.genes), top_fraction=args.top_fraction,
        criterion="loo", ad_modes=AD_MODES, sweep=True,
    )
    d = res.to_dict()

    print(f"\nrecommended: {d['recommended']}   (n edges = {len(d['edges'])})")
    print("\nfull ranking (top 12):")
    for i, r in enumerate(d["ranking"][:12]):
        p = ",".join(f"{k}={v:g}" for k, v in sorted(r["params"].items()))
        print(f"  {i+1:2d}. {r['method']:<20}{p:<24}{r['loo_nll']:.3f}")

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
        show = label
        lam, alpha = r["params"].get("lam"), r["params"].get("alpha")
        if method == "ad_target_ridge" and lam is not None:
            show = f"AD-Ridge (Eq. 2, $\\lambda={lam:g}$)"
        elif method == "ad_ridge" and alpha is not None:
            show = f"AD-Ridge, hard projection ($\\lambda=1$, $\\alpha={alpha:g}$)"
        elif method == "ridge" and alpha is not None:
            show = f"{label} ($\\alpha={alpha:g}$)"
        rows.append((rank, show, r["loo_nll"]))
    rows.sort(key=lambda t: t[2])

    tex = ["\\begin{tabular}{clr}", "\\toprule",
           "Rank & Method & Mean LOO-NLL\\\\", "\\midrule"]
    for i, (_, label, nll) in enumerate(rows, start=1):
        tex.append(f"{i} & {label} & {nll:.3f}\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(tex)
    print("\n--- Table 1 (LaTeX) ---\n" + table)

    # ---- Derived outputs (Data Availability) -------------------------------
    print("\nderived outputs:")
    od = args.outdir
    with open(os.path.join(od, "table1.tex"), "w") as fh:
        fh.write(table + "\n")
    print("  wrote table1.tex")

    genes = d["genes"]
    write_csv(os.path.join(od, "gene_groups.csv"), ["gene", "group", "group_id"],
              [(g, n, c) for g, n, c in zip(genes, names, codes)])

    write_csv(os.path.join(od, "estimator_ranking.csv"),
              ["rank", "method", "params", "loo_nll", "condition_number"],
              [(i, r["method"],
                ";".join(f"{k}={v:g}" for k, v in sorted(r["params"].items())),
                f"{r['loo_nll']:.6f}", f"{r['condition_number']:.6f}")
               for i, r in enumerate(d["ranking"], 1)])

    cov = np.asarray(d["covariance"], dtype=float)
    with open(os.path.join(od, "best_covariance.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + list(genes))
        for g, row in zip(genes, cov):
            w.writerow([g] + [f"{v:.10g}" for v in row])
    print("  wrote best_covariance.csv")

    write_csv(os.path.join(od, "top_edges.csv"),
              ["gene_a", "gene_b", "covariance", "abs_covariance"],
              [(e["gene_a"], e["gene_b"], f"{e['covariance']:.10g}",
                f"{e['abs_covariance']:.10g}") for e in d["edges"]])

    comm = louvain_communities(d["edges"], genes)
    if comm is not None:
        write_csv(os.path.join(od, "communities.csv"), ["gene", "community"],
                  [(g, comm[g]) for g in genes if g in comm])
    else:
        print("  (networkx unavailable; communities.csv skipped)")

    payload = {"grouping": args.group, "group_sizes": sizes,
               "groups": dict(Counter(names)), "n_samples": int(X.shape[0]),
               "n_genes": int(X.shape[1]), **d}
    with open(os.path.join(od, "gse52778_result.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print("  wrote gse52778_result.json")

    with open(os.path.join(od, "report.md"), "w") as fh:
        fh.write(f"# ADGENCOV: GSE52778 run summary\n\n")
        fh.write(f"- matrix: {X.shape[0]} samples x {X.shape[1]} genes\n")
        fh.write(f"- symmetry: `{args.group}`, group sizes {sizes}\n")
        fh.write(f"- criterion: exact leave-one-out Gaussian NLL\n")
        fh.write(f"- candidates scored: {len(d['ranking'])}\n")
        fh.write(f"- recommended: `{d['recommended']}`\n\n")
        fh.write("| Rank | Method | Params | LOO-NLL |\n|---:|---|---|---:|\n")
        for i, r in enumerate(d["ranking"][:12], 1):
            p = ", ".join(f"{k}={v:g}" for k, v in sorted(r["params"].items())) or "--"
            fh.write(f"| {i} | `{r['method']}` | {p} | {r['loo_nll']:.3f} |\n")
    print("  wrote report.md")
    print(f"\nall outputs in {od}")


if __name__ == "__main__":
    main()
