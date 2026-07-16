"""Render the covariance-network figure from an analysis payload.

The manuscript figure and the dashboard's "Download figure" button call
:func:`render_network`, so the figure in the paper and the figure a user
downloads are produced by the same code and cannot drift apart.

Deliberately built on ``matplotlib.figure.Figure`` + ``FigureCanvasAgg`` rather
than ``pyplot``: pyplot keeps a global figure registry and is not thread-safe,
and this renders inside a web request thread.

Every label is derived from the payload.  Community labels state each module's
actual composition rather than asserting a biological role, and the caption says
"covariance" because that is what the estimator computes.

Requires the ``figure`` extra (matplotlib, networkx).
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Sequence

COMMUNITY_COLORS = ["#2f8fe0", "#f0555f", "#39b57a", "#f0b429", "#9b5de5",
                    "#f97316", "#0ea5e9", "#ec4899", "#84cc16", "#14b8a6"]
EDGE_RAMP = ["#9dc3ea", "#7d9fe0", "#8f6fd0", "#5b21b6"]
EDGE_WIDTH = [0.35, 0.7, 1.3, 2.4]
EDGE_ALPHA = [0.30, 0.42, 0.62, 0.90]
STRENGTH = ["Weak", "Moderate", "Strong", "Very strong"]
MARKER = {"protein_coding": "o", "miRNA": "s", "snoRNA": "D"}
CLASS_LABEL = {"protein_coding": "Protein-coding gene", "miRNA": "miRNA", "snoRNA": "snoRNA"}
INK, MUTED, RULE = "#10161f", "#5b6675", "#c9d3e0"


class FigureUnavailable(RuntimeError):
    """Raised when the plotting extra is not installed."""


def rna_class(gene: str) -> str:
    g = str(gene).upper()
    if re.match(r"^(HSA-)?(MIR|LET-?7)", g):
        return "miRNA"
    if re.match(r"^(SNORD|SNORA|SCARNA)", g):
        return "snoRNA"
    return "protein_coding"


def _bucket(t: float) -> int:
    return 3 if t >= 0.66 else 2 if t >= 0.40 else 1 if t >= 0.18 else 0


def community_label(i: int, members: Sequence[str]) -> str:
    """Describe a module by what it contains — never by an assumed function."""
    order = ("protein_coding", "miRNA", "snoRNA")
    pretty = {"protein_coding": "protein-coding", "miRNA": "miRNA", "snoRNA": "snoRNA"}
    classes = [rna_class(m) for m in members]
    parts = [f"{classes.count(c)} {pretty[c]}" for c in order if classes.count(c)]
    return f"C{i + 1}: " + " · ".join(parts)


def _modular_layout(nx, G, comms, seed=7):
    """Communities as separated blobs: centroids on a circle, spring inside."""
    import numpy as np

    pos, k = {}, len(comms)
    R = 1.0 if k > 1 else 0.0
    biggest = max((len(c) for c in comms), default=1)
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
        scale = 0.42 * (0.6 + 0.4 * (len(comp) / biggest) ** 0.5)
        for name, q in p.items():
            pos[name] = (cx + (q[0] / sx) * scale * 2, cy + (q[1] / sy) * scale * 2)
    return pos


def render_network(
    payload: Dict[str, Any],
    *,
    fmt: str = "pdf",
    top_pct: Optional[float] = None,
    dataset: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> bytes:
    """Render the AD covariance network for *payload* and return the file bytes.

    *top_pct* defaults to the edge budget implied by the payload itself
    (edges / all gene pairs), so the figure always describes what it shows.
    """
    try:
        import matplotlib
        import networkx as nx
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyBboxPatch
        import matplotlib.patheffects as pe
    except Exception as exc:  # noqa: BLE001
        raise FigureUnavailable(
            "figure rendering needs the plotting extra: pip install 'adgencov[figure]'"
            f" ({exc})"
        ) from exc

    edges = payload.get("edges") or []
    if not edges:
        raise ValueError("this result has no covariance edges to draw")

    genes = payload.get("genes") or []
    p = payload.get("n_genes") or len(genes)
    if top_pct is None and p and p > 1:
        top_pct = 100.0 * len(edges) / (p * (p - 1) / 2.0)

    G = nx.Graph()
    for e in edges:
        G.add_edge(e["gene_a"], e["gene_b"], cov=e["covariance"], w=abs(e["covariance"]))

    comms = sorted(nx.community.louvain_communities(G, weight="w", seed=0),
                   key=len, reverse=True)
    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    deg = dict(G.degree())
    maxdeg = max(deg.values()) or 1
    maxabs = max(abs(e["covariance"]) for e in edges) or 1.0
    pos = _modular_layout(nx, G, comms)

    fig = Figure(figsize=(15.4, 10.2), dpi=150)
    FigureCanvasAgg(fig)                     # no pyplot: thread-safe
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 0.70, 0.905])
    ax.axis("off")

    for u, v, a in sorted(G.edges(data=True), key=lambda t: abs(t[2]["cov"])):
        b = _bucket(abs(a["cov"]) / maxabs)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=EDGE_RAMP[b],
                linewidth=EDGE_WIDTH[b], alpha=EDGE_ALPHA[b], zorder=1,
                solid_capstyle="round")

    for cls in ("protein_coding", "miRNA", "snoRNA"):
        ns = [n for n in G.nodes if rna_class(n) == cls]
        if not ns:
            continue
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns],
                   s=[60 + 900 * (deg[n] / maxdeg) ** 1.15 for n in ns],
                   c=[COMMUNITY_COLORS[comm_of[n] % len(COMMUNITY_COLORS)] for n in ns],
                   marker=MARKER[cls], edgecolors="white", linewidths=1.0, zorder=3)

    hubs = sorted(deg, key=lambda n: -deg[n])
    hub_set = set(hubs[:8])
    for n in G.nodes:
        big = n in hub_set
        ax.annotate(str(n), pos[n], fontsize=8.6 if big else 5.9, ha="center",
                    va="center", zorder=4, fontweight="bold" if big else "normal",
                    color=INK if big else "#33404f",
                    xytext=(0, 10 + 0.012 * (60 + 900 * (deg[n] / maxdeg) ** 1.15) ** 0.5),
                    textcoords="offset points",
                    path_effects=[pe.withStroke(linewidth=2.6 if big else 2.0,
                                                foreground="white")])
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.65, 1.65)

    method = payload.get("recommended", "AD")
    nice = {"ad_target_ridge": "AD-Ridge", "ad_target_lw": "AD-Ledoit-Wolf",
            "ad_target_oas": "AD-OAS", "ad_ridge": "AD-Ridge (hard projection)",
            "ad_linear_lw": "AD-Ledoit-Wolf (hard projection)",
            "ad_oas": "AD-OAS (hard projection)", "oas": "OAS", "lw": "Ledoit-Wolf",
            "ad_lasso": "AD-LASSO", "ad_elastic_net": "AD-Elastic-Net"}.get(method, method)
    t = fig.text(0.012, 0.965, f"{nice} Covariance Network", fontsize=21,
                 fontweight="bold", color=INK, va="top")
    fig.canvas.draw()
    tw = t.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()).width
    fig.text(0.012 + tw + 0.010, 0.9585,
             f"(Top {top_pct:.3g}% Strongest Connections)" if top_pct else "",
             fontsize=13.5, color=MUTED, va="top")
    src = payload.get("source") or {}
    ds = dataset or src.get("accession") or ("+".join(src.get("accessions", []))
                                             or src.get("kind", "analysis"))
    sub = subtitle or ""
    n_samp = (payload.get("combined") or {}).get("n_samples_total") or payload.get("n_samples")
    bits = [b for b in [ds, sub] if b]
    tail = f"({n_samp} samples, {p} genes analysed)" if n_samp else f"({p} genes analysed)"
    fig.text(0.012, 0.928, " — ".join(bits) + " " + tail, fontsize=11.5,
             color=MUTED, style="italic", va="top")

    lx, lw = 0.715, 0.275
    fx = fig.transFigure

    def fpanel(x, y, w, h):
        fig.patches.append(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.9, edgecolor=RULE, facecolor="white", transform=fx, zorder=5))

    fpanel(lx, 0.775, lw, 0.185)
    fig.text(lx + 0.012, 0.943, "Node type", fontsize=10, fontweight="bold",
             color=INK, va="top", zorder=6)
    for i, cls in enumerate(("protein_coding", "miRNA", "snoRNA")):
        yy = 0.915 - i * 0.031
        fig.text(lx + 0.030, yy, CLASS_LABEL[cls], fontsize=9, color=INK,
                 va="center", zorder=6)
        fig.lines.append(Line2D([lx + 0.019], [yy], marker=MARKER[cls], color="none",
                                markerfacecolor="white", markeredgecolor=INK,
                                markeredgewidth=1.1, markersize=8, transform=fx, zorder=6))
    fig.text(lx + 0.148, 0.943, "Edge strength", fontsize=10, fontweight="bold",
             color=INK, va="top", zorder=6)
    for i in range(4):
        yy = 0.915 - i * 0.031
        fig.lines.append(Line2D([lx + 0.155, lx + 0.196], [yy, yy], color=EDGE_RAMP[i],
                                linewidth=EDGE_WIDTH[i] + 0.9,
                                alpha=max(EDGE_ALPHA[i], 0.75), transform=fx, zorder=6))
        fig.text(lx + 0.204, yy, STRENGTH[i], fontsize=9, color=INK, va="center", zorder=6)

    ch = 0.075 + 0.030 * len(comms)
    cy0 = 0.755 - ch
    fpanel(lx, cy0, lw, ch)
    fig.text(lx + 0.012, cy0 + ch - 0.018, "Community (Louvain)", fontsize=10,
             fontweight="bold", color=INK, va="top", zorder=6)
    for i, c in enumerate(comms):
        yy = cy0 + ch - 0.048 - i * 0.030
        fig.lines.append(Line2D([lx + 0.020], [yy], marker="o", color="none",
                                markerfacecolor=COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)],
                                markeredgecolor="white", markersize=9, transform=fx, zorder=6))
        fig.text(lx + 0.034, yy, community_label(i, sorted(c)), fontsize=8.6,
                 color=INK, va="center", zorder=6)
        fig.text(lx + lw - 0.014, yy, f"n = {len(c)}", fontsize=9, color=MUTED,
                 va="center", ha="right", zorder=6)
    fig.text(lx + 0.012, cy0 + 0.020,
             f"Total nodes = {G.number_of_nodes()}   ·   Edges shown = {G.number_of_edges()}"
             + (f" (top {top_pct:.3g}%)" if top_pct else ""),
             fontsize=9.2, color=INK, va="center", zorder=6)

    hh = 0.075 + 0.028 * min(8, len(hubs))
    hy0 = cy0 - 0.030 - hh
    fpanel(lx, hy0, lw, hh)
    fig.text(lx + 0.012, hy0 + hh - 0.018, "Top hub nodes by degree", fontsize=10,
             fontweight="bold", color=INK, va="top", zorder=6)
    for lbl, dx, ha in (("Node", 0.034, "left"), ("Type", 0.135, "left"),
                        ("Degree", lw - 0.014, "right")):
        fig.text(lx + dx, hy0 + hh - 0.046, lbl, fontsize=8.6, fontweight="bold",
                 color=MUTED, va="center", ha=ha, zorder=6)
    for i, n in enumerate(hubs[:8]):
        yy = hy0 + hh - 0.074 - i * 0.028
        cls = rna_class(n)
        # Same shape + community colour as the graph, by construction.
        fig.lines.append(Line2D([lx + 0.020], [yy], marker=MARKER[cls], color="none",
                                markerfacecolor=COMMUNITY_COLORS[comm_of[n] % len(COMMUNITY_COLORS)],
                                markeredgecolor="white", markersize=7.5, transform=fx, zorder=6))
        fig.text(lx + 0.034, yy, str(n), fontsize=8.6, color=INK, va="center", zorder=6)
        fig.text(lx + 0.135, yy, CLASS_LABEL[cls].replace(" gene", ""), fontsize=8.6,
                 color=INK, va="center", zorder=6)
        fig.text(lx + lw - 0.014, yy, str(deg[n]), fontsize=8.6, color=INK,
                 va="center", ha="right", zorder=6)

    import textwrap
    cap = (f"Edges represent covariance estimated by {nice}. Edge width and colour "
           "intensity are proportional to |covariance|; node size is proportional to "
           "degree.")
    if top_pct:
        cap += f" Only the strongest {top_pct:.3g}% of gene pairs are shown."
    lines = textwrap.wrap(cap, width=52)
    cap_h = 0.028 + 0.017 * len(lines)
    cap_y = hy0 - 0.028 - cap_h
    fig.patches.append(FancyBboxPatch((lx, cap_y), lw, cap_h,
                                      boxstyle="round,pad=0.004,rounding_size=0.008",
                                      linewidth=0.9, edgecolor="#bcd3ea",
                                      facecolor="#f4f9ff", transform=fx, zorder=5))
    fig.text(lx + 0.014, cap_y + cap_h - 0.016, "\n".join(lines), fontsize=8.4,
             color="#2c4a63", style="italic", va="top", zorder=6, linespacing=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, facecolor="white", dpi=150)
    return buf.getvalue()
