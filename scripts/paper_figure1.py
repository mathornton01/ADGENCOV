#!/usr/bin/env python3
"""Render the AD-Ridge covariance network figure from real ADGENCOV output.

Publication-grade rendering of the JSON emitted by scripts/paper_gse52778.py:
communities laid out as separated modules, nodes coloured by Louvain community
and shaped by molecular class, sized by degree; edges width/colour scaled by
|covariance|.  Ships the full legend set (node type, edge strength, community
sizes, totals), a hub table, and a caption box, and writes vector PDF + JPEG.

Every label is derived from the data.  Community names are composed from each
module's actual composition (dominant molecular class), never asserted.

Usage:
    python3 scripts/paper_figure1.py --result paper_out20/gse52778_result.json \
        --out paper_out20/fig_network_ad_ridge --top-pct 20
"""
from __future__ import annotations

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# Saturated, well-separated community colours (colour-blind leaning).
COMMUNITY_COLORS = ["#2f8fe0", "#f0555f", "#39b57a", "#f0b429", "#9b5de5",
                    "#f97316", "#0ea5e9", "#ec4899", "#84cc16", "#14b8a6"]
# Edge ramp: weak = pale blue -> very strong = deep purple (as in the note).
EDGE_RAMP = ["#9dc3ea", "#7d9fe0", "#8f6fd0", "#5b21b6"]
EDGE_WIDTH = [0.35, 0.7, 1.3, 2.4]
EDGE_ALPHA = [0.30, 0.42, 0.62, 0.90]
STRENGTH = ["Weak", "Moderate", "Strong", "Very strong"]

MARKER = {"protein_coding": "o", "miRNA": "s", "snoRNA": "D"}
CLASS_LABEL = {"protein_coding": "Protein-coding gene", "miRNA": "miRNA", "snoRNA": "snoRNA"}
INK, MUTED, RULE = "#10161f", "#5b6675", "#c9d3e0"


def rna_class(gene: str) -> str:
    g = gene.upper()
    if re.match(r"^(HSA-)?(MIR|LET-?7)", g):
        return "miRNA"
    if re.match(r"^(SNORD|SNORA|SCARNA)", g):
        return "snoRNA"
    return "protein_coding"


def bucket(t: float) -> int:
    return 3 if t >= 0.66 else 2 if t >= 0.40 else 1 if t >= 0.18 else 0


def community_name(i, members) -> str:
    """Label a module by its actual composition — never by an assumed function."""
    order = ("protein_coding", "miRNA", "snoRNA")
    pretty = {"protein_coding": "protein-coding", "miRNA": "miRNA", "snoRNA": "snoRNA"}
    classes = [rna_class(m) for m in members]
    parts = [f"{classes.count(c)} {pretty[c]}" for c in order if classes.count(c)]
    return f"C{i + 1}: " + " · ".join(parts)


def modular_layout(G, comms, seed=7):
    """Lay communities out as separated blobs: centroids on a circle, spring inside."""
    pos = {}
    k = len(comms)
    R = 1.0 if k > 1 else 0.0
    for i, comp in enumerate(comms):
        ang = 2 * np.pi * i / max(1, k)
        cx, cy = R * np.cos(ang), R * np.sin(ang)
        sub = G.subgraph(comp)
        if sub.number_of_nodes() == 1:
            pos[next(iter(comp))] = (cx, cy)
            continue
        p = nx.spring_layout(sub, seed=seed, weight="w", k=0.55, iterations=400)
        xs = [q[0] for q in p.values()]
        ys = [q[1] for q in p.values()]
        sx = (max(xs) - min(xs)) or 1.0
        sy = (max(ys) - min(ys)) or 1.0
        scale = 0.42 * (0.6 + 0.4 * np.sqrt(len(comp) / max(1, max(len(c) for c in comms))))
        for nname, q in p.items():
            pos[nname] = (cx + (q[0] / sx) * scale * 2, cy + (q[1] / sy) * scale * 2)
    return pos


def panel(ax, x, y, w, h, title=None):
    """A rounded card, matching the note's boxed legend blocks."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=0.9, edgecolor=RULE, facecolor="white",
                                transform=ax.transAxes, zorder=5, clip_on=False))
    if title:
        ax.text(x + 0.012, y + h - 0.022, title, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=INK, zorder=6, va="top")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="paper_out20/gse52778_result.json")
    ap.add_argument("--out", default="paper_out20/fig_network_ad_ridge")
    ap.add_argument("--top-pct", type=float, default=20.0)
    ap.add_argument("--dataset", default="GSE52778")
    ap.add_argument("--subtitle", default="airway smooth muscle RNA-seq")
    args = ap.parse_args()

    d = json.load(open(args.result))
    edges, n_analyzed = d["edges"], d["n_genes"]
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["gene_a"], e["gene_b"], cov=e["covariance"], w=abs(e["covariance"]))

    comms = sorted(nx.community.louvain_communities(G, weight="w", seed=0),
                   key=len, reverse=True)
    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    deg = dict(G.degree())
    maxdeg = max(deg.values()) or 1
    maxabs = max(abs(e["covariance"]) for e in edges) or 1.0
    pos = modular_layout(G, comms)

    fig = plt.figure(figsize=(15.4, 10.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 0.70, 0.905]); ax.axis("off")
    ax.set_facecolor("white")

    # ---- edges (weak first so strong ones read on top) --------------------
    for u, v, a in sorted(G.edges(data=True), key=lambda t: abs(t[2]["cov"])):
        b = bucket(abs(a["cov"]) / maxabs)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=EDGE_RAMP[b],
                linewidth=EDGE_WIDTH[b], alpha=EDGE_ALPHA[b], zorder=1,
                solid_capstyle="round")

    # ---- nodes -------------------------------------------------------------
    for cls in ("protein_coding", "miRNA", "snoRNA"):
        ns = [n for n in G.nodes if rna_class(n) == cls]
        if not ns:
            continue
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns],
                   s=[60 + 900 * (deg[n] / maxdeg) ** 1.15 for n in ns],
                   c=[COMMUNITY_COLORS[comm_of[n] % len(COMMUNITY_COLORS)] for n in ns],
                   marker=MARKER[cls], edgecolors="white", linewidths=1.0, zorder=3)

    # Hubs get a larger label; the rest stay small so the plot keeps breathing.
    hubs = sorted(deg, key=lambda n: -deg[n])
    hub_set = set(hubs[:8])
    for n in G.nodes:
        big = n in hub_set
        ax.annotate(n, pos[n], fontsize=8.6 if big else 5.9, ha="center", va="center",
                    zorder=4, fontweight="bold" if big else "normal",
                    color=INK if big else "#33404f",
                    xytext=(0, 10 + 0.012 * (60 + 900 * (deg[n] / maxdeg) ** 1.15) ** 0.5),
                    textcoords="offset points",
                    path_effects=[pe.withStroke(linewidth=2.6 if big else 2.0,
                                                foreground="white")])
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.65, 1.65)

    # ---- titles ------------------------------------------------------------
    t = fig.text(0.012, 0.965, "AD-Ridge Covariance Network", fontsize=21,
                 fontweight="bold", color=INK, va="top")
    fig.canvas.draw()  # realize the title so its width can be measured
    tw = t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).width
    fig.text(0.012 + tw + 0.010, 0.9585,
             f"(Top {args.top_pct:g}% Strongest Connections)",
             fontsize=13.5, color=MUTED, va="top")
    fig.text(0.012, 0.928,
             f"{args.dataset} — {args.subtitle} (n = {d.get('n_samples','?')} samples, "
             f"{n_analyzed} genes analysed)",
             fontsize=11.5, color=MUTED, style="italic", va="top")

    # ---- right-hand legend column -----------------------------------------
    lx, lw = 0.715, 0.275
    fx = fig.transFigure

    def fpanel(x, y, w, h, title=None):
        fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                                          linewidth=0.9, edgecolor=RULE, facecolor="white",
                                          transform=fx, zorder=5))
        if title:
            fig.text(x + 0.012, y + h - 0.018, title, fontsize=10, fontweight="bold",
                     color=INK, zorder=6, va="top")

    # node type + edge strength
    fpanel(lx, 0.775, lw, 0.185)
    fig.text(lx + 0.012, 0.943, "Node type", fontsize=10, fontweight="bold", color=INK, va="top", zorder=6)
    for i, cls in enumerate(("protein_coding", "miRNA", "snoRNA")):
        yy = 0.915 - i * 0.031
        fig.text(lx + 0.030, yy, CLASS_LABEL[cls], fontsize=9, color=INK, va="center", zorder=6)
        fig.lines.append(Line2D([lx + 0.019], [yy], marker=MARKER[cls], color="none",
                                markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.1,
                                markersize=8, transform=fx, zorder=6))
    fig.text(lx + 0.148, 0.943, "Edge strength", fontsize=10, fontweight="bold", color=INK, va="top", zorder=6)
    for i in range(4):
        yy = 0.915 - i * 0.031
        fig.lines.append(Line2D([lx + 0.155, lx + 0.196], [yy, yy], color=EDGE_RAMP[i],
                                linewidth=EDGE_WIDTH[i] + 0.9, alpha=max(EDGE_ALPHA[i], 0.75),
                                transform=fx, zorder=6))
    for i in range(4):
        fig.text(lx + 0.204, 0.915 - i * 0.031, STRENGTH[i], fontsize=9, color=INK,
                 va="center", zorder=6)

    # communities
    ch = 0.075 + 0.030 * len(comms)
    cy0 = 0.755 - ch
    fpanel(lx, cy0, lw, ch, "Community (Louvain)")
    for i, c in enumerate(comms):
        yy = cy0 + ch - 0.048 - i * 0.030
        fig.lines.append(Line2D([lx + 0.020], [yy], marker="o", color="none",
                                markerfacecolor=COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)],
                                markeredgecolor="white", markersize=9, transform=fx, zorder=6))
        fig.text(lx + 0.034, yy, community_name(i, c), fontsize=8.6, color=INK, va="center", zorder=6)
        fig.text(lx + lw - 0.014, yy, f"n = {len(c)}", fontsize=9, color=MUTED,
                 va="center", ha="right", zorder=6)
    fig.text(lx + 0.012, cy0 + 0.020, f"Total nodes = {G.number_of_nodes()}   ·   "
             f"Edges shown = {G.number_of_edges()} (top {args.top_pct:g}%)",
             fontsize=9.2, color=INK, va="center", zorder=6)

    # hub table
    hh = 0.075 + 0.028 * min(8, len(hubs))
    hy0 = cy0 - 0.030 - hh
    fpanel(lx, hy0, lw, hh, "Top hub nodes by degree")
    fig.text(lx + 0.034, hy0 + hh - 0.046, "Node", fontsize=8.6, fontweight="bold", color=MUTED, va="center", zorder=6)
    fig.text(lx + 0.135, hy0 + hh - 0.046, "Type", fontsize=8.6, fontweight="bold", color=MUTED, va="center", zorder=6)
    fig.text(lx + lw - 0.014, hy0 + hh - 0.046, "Degree", fontsize=8.6, fontweight="bold",
             color=MUTED, va="center", ha="right", zorder=6)
    for i, n in enumerate(hubs[:8]):
        yy = hy0 + hh - 0.074 - i * 0.028
        cls = rna_class(n)
        # marker matches the graph exactly: same shape for class, same community colour
        fig.lines.append(Line2D([lx + 0.020], [yy], marker=MARKER[cls], color="none",
                                markerfacecolor=COMMUNITY_COLORS[comm_of[n] % len(COMMUNITY_COLORS)],
                                markeredgecolor="white", markersize=7.5, transform=fx, zorder=6))
        fig.text(lx + 0.034, yy, n, fontsize=8.6, color=INK, va="center", zorder=6)
        fig.text(lx + 0.135, yy, CLASS_LABEL[cls].replace(" gene", ""), fontsize=8.6,
                 color=INK, va="center", zorder=6)
        fig.text(lx + lw - 0.014, yy, str(deg[n]), fontsize=8.6, color=INK, va="center",
                 ha="right", zorder=6)

    # caption box — states exactly what the estimator computed
    cap = ("Edges represent covariance estimated by AD-Ridge (Eq. 2 convex form). "
           "Edge width and colour intensity are proportional to |covariance|; node size is "
           f"proportional to degree. Only the strongest {args.top_pct:g}% of gene pairs are shown.")
    import textwrap
    cap_lines = textwrap.wrap(cap, width=52)
    cap_h = 0.028 + 0.017 * len(cap_lines)
    cap_y = hy0 - 0.028 - cap_h
    fig.patches.append(FancyBboxPatch((lx, cap_y), lw, cap_h,
                                      boxstyle="round,pad=0.004,rounding_size=0.008",
                                      linewidth=0.9, edgecolor="#bcd3ea", facecolor="#f4f9ff",
                                      transform=fx, zorder=5))
    fig.text(lx + 0.014, cap_y + cap_h - 0.016, "\n".join(cap_lines), fontsize=8.4,
             color="#2c4a63", style="italic", va="top", zorder=6, linespacing=1.5)

    for ext in ("pdf", "jpeg"):
        fig.savefig(f"{args.out}.{ext}", facecolor="white", dpi=150)
        print("wrote", f"{args.out}.{ext}")


if __name__ == "__main__":
    main()
