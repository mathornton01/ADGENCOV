#!/usr/bin/env python3
"""Render Figure 1 (the AD-Ridge covariance network) from real ADGENCOV output.

Reads the JSON emitted by scripts/paper_gse52778.py and draws the top-covariance
network: nodes coloured by Louvain community, shaped by molecular class, sized by
degree; edges width/alpha scaled by |covariance|.  Emits both PDF (vector, for
the manuscript) and JPEG.

Usage:
    python3 scripts/paper_figure1.py --result paper_out/gse52778_result.json \
                                     --out paper_out/fig_network_top20_ad_ridge
"""
from __future__ import annotations

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
from matplotlib.lines import Line2D

COMMUNITY_COLORS = ["#4aa3ff", "#ff6b7d", "#5ed19b", "#f5c451", "#b98cff",
                    "#ff9f56", "#5fd0e0", "#f78fb3", "#a3d65c", "#ff7ac0"]


def rna_class(gene: str) -> str:
    g = gene.upper()
    if re.match(r"^(HSA-)?(MIR|LET-?7)", g):
        return "miRNA"
    if re.match(r"^(SNORD|SNORA|SCARNA)", g):
        return "snoRNA"
    return "protein_coding"


MARKER = {"protein_coding": "o", "miRNA": "s", "snoRNA": "D"}
CLASS_LABEL = {"protein_coding": "Protein-coding gene", "miRNA": "miRNA", "snoRNA": "snoRNA"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="paper_out/gse52778_result.json")
    ap.add_argument("--out", default="paper_out/fig_network_top20_ad_ridge")
    args = ap.parse_args()

    d = json.load(open(args.result))
    edges = d["edges"]
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["gene_a"], e["gene_b"], cov=e["covariance"], w=abs(e["covariance"]))

    comms = nx.community.louvain_communities(G, weight="w", seed=0)
    comms = sorted(comms, key=len, reverse=True)
    comm_of = {n: i for i, c in enumerate(comms) for n in c}

    deg = dict(G.degree())
    maxdeg = max(deg.values()) or 1
    maxabs = max(abs(e["covariance"]) for e in edges) or 1.0

    # Lay out each connected component separately, then pack them side by side.
    # A plain spring_layout flings small disconnected components to the margins
    # and leaves most of the canvas empty.
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    pos, x_off = {}, 0.0
    for comp in comps:
        sub = G.subgraph(comp)
        p = nx.spring_layout(sub, seed=7, weight="w", k=1.1, iterations=500)
        xs = [q[0] for q in p.values()] or [0.0]
        ys = [q[1] for q in p.values()] or [0.0]
        span_x = (max(xs) - min(xs)) or 0.6
        span_y = (max(ys) - min(ys)) or 0.6
        for n, q in p.items():
            pos[n] = ((q[0] - min(xs)) / span_x + x_off, (q[1] - min(ys)) / span_y)
        x_off += 1.25

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=300)

    for u, v, a in G.edges(data=True):
        t = abs(a["cov"]) / maxabs
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#6a30b8" if a["cov"] >= 0 else "#2f52b8",
                linewidth=0.6 + 3.0 * t, alpha=0.25 + 0.6 * t, zorder=1,
                solid_capstyle="round")

    for cls in ("protein_coding", "miRNA", "snoRNA"):
        ns = [n for n in G.nodes if rna_class(n) == cls]
        if not ns:
            continue
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns],
                   s=[70 + 620 * (deg[n] / maxdeg) for n in ns],
                   c=[COMMUNITY_COLORS[comm_of[n] % len(COMMUNITY_COLORS)] for n in ns],
                   marker=MARKER[cls], edgecolors="white", linewidths=0.7, zorder=2)

    # Labels sit just above each node with a white halo so they stay readable
    # over edges and markers.
    for n in G.nodes:
        r = 70 + 620 * (deg[n] / maxdeg)
        ax.annotate(n, (pos[n][0], pos[n][1]), fontsize=6.0, ha="center",
                    va="center", zorder=4, fontweight="bold", color="#10161f",
                    xytext=(0, 7 + 0.010 * r), textcoords="offset points",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    shape_h = [Line2D([], [], marker=MARKER[c], color="none", markerfacecolor="#9aa7b8",
                      markeredgecolor="white", markersize=8, label=CLASS_LABEL[c])
               for c in ("protein_coding", "miRNA", "snoRNA")]
    comm_h = [Line2D([], [], marker="o", color="none",
                     markerfacecolor=COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)],
                     markeredgecolor="white", markersize=8,
                     label=f"Community {i+1} (n = {len(c)})")
              for i, c in enumerate(comms)]
    edge_h = [Line2D([], [], color="#6a30b8", linewidth=0.6 + 3.0 * f,
                     alpha=0.25 + 0.6 * f, label=lab)
              for f, lab in [(0.15, "Weak"), (0.45, "Moderate"), (0.75, "Strong"), (1.0, "Very strong")]]

    l1 = ax.legend(handles=shape_h, title="Node type", loc="upper left",
                   bbox_to_anchor=(1.01, 1.0), fontsize=7, title_fontsize=7.5, frameon=False)
    l2 = ax.legend(handles=comm_h, title="Community (Louvain)", loc="upper left",
                   bbox_to_anchor=(1.01, 0.74), fontsize=7, title_fontsize=7.5, frameon=False)
    ax.add_artist(l1)
    ax.legend(handles=edge_h, title="Edge strength (|covariance|)", loc="upper left",
              bbox_to_anchor=(1.01, 0.40), fontsize=7, title_fontsize=7.5, frameon=False)
    ax.add_artist(l2)

    ax.set_title("AD-Ridge covariance network — GSE52778 (top 1% of edges)",
                 fontsize=9.5, pad=8)
    ax.text(0.5, -0.04, f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges · "
            f"node size ∝ degree", transform=ax.transAxes, ha="center", fontsize=7, color="#5b6675")
    ax.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "jpeg"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=300)
        print("wrote", f"{args.out}.{ext}")


if __name__ == "__main__":
    main()
