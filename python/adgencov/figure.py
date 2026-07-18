"""Render the covariance-network figure from an analysis payload.

The manuscript figure and the dashboard's "Download figure" button both call
:func:`render_network`, so the figure in the paper and the one a user downloads
are produced by the same code and cannot drift apart.

Everything is driven by :class:`FigureConfig`, so callers can retheme and
re-lay-out the figure without editing this module (see ``FigureConfig`` for the
knobs; the CLI and the HTTP export endpoint both expose them).

Built on ``matplotlib.figure.Figure`` + ``FigureCanvasAgg`` rather than
``pyplot``: pyplot keeps a global figure registry and is not thread-safe, and
this renders inside a web request thread.

Every label is derived from the payload.  Community labels state each module's
actual composition rather than asserting a biological role, and the caption says
"covariance" because that is what the estimator computes.

Requires the ``figure`` extra (matplotlib, networkx).
"""
from __future__ import annotations

import io
import re
import textwrap
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

COMMUNITY_COLORS: Tuple[str, ...] = (
    "#2f8fe0", "#f0555f", "#39b57a", "#f0b429", "#9b5de5",
    "#f97316", "#0ea5e9", "#ec4899", "#84cc16", "#14b8a6",
)
EDGE_RAMP: Tuple[str, ...] = ("#9dc3ea", "#7d9fe0", "#8f6fd0", "#5b21b6")
EDGE_WIDTHS: Tuple[float, ...] = (0.35, 0.7, 1.3, 2.4)
EDGE_ALPHAS: Tuple[float, ...] = (0.30, 0.42, 0.62, 0.90)
STRENGTH_LABELS: Tuple[str, ...] = ("Weak", "Moderate", "Strong", "Very strong")
MARKER = {"protein_coding": "o", "miRNA": "s", "snoRNA": "D"}
CLASS_LABEL = {"protein_coding": "Protein-coding gene", "miRNA": "miRNA", "snoRNA": "snoRNA"}


@dataclass
class FigureConfig:
    """Every knob the network figure exposes.

    Pass one to :func:`render_network`, or override individual fields with
    keyword arguments.  Nothing here changes what is *measured* — only how it is
    drawn — so a retheme can never alter the reported numbers.
    """

    # -- canvas ------------------------------------------------------------
    figsize: Tuple[float, float] = (15.4, 10.2)
    dpi: int = 150
    facecolor: str = "white"
    ink: str = "#10161f"
    muted: str = "#5b6675"
    rule: str = "#c9d3e0"

    # -- palette -----------------------------------------------------------
    community_colors: Sequence[str] = COMMUNITY_COLORS
    edge_ramp: Sequence[str] = EDGE_RAMP
    edge_widths: Sequence[float] = EDGE_WIDTHS
    edge_alphas: Sequence[float] = EDGE_ALPHAS
    edge_thresholds: Sequence[float] = (0.18, 0.40, 0.66)   # weak|mod|strong|very

    # -- nodes -------------------------------------------------------------
    node_size_min: float = 60.0
    node_size_span: float = 900.0
    node_size_gamma: float = 1.15
    node_edge_color: str = "white"

    # -- labels ------------------------------------------------------------
    label_nodes: bool = True
    max_labels: Optional[int] = None      # None = label every node
    hub_label_count: int = 8              # how many get the larger, bold label
    label_size: float = 5.9
    hub_label_size: float = 8.6
    declutter: bool = True                # push overlapping labels apart
    declutter_iters: int = 220

    # -- layout ------------------------------------------------------------
    layout_seed: int = 7
    layout_iterations: int = 400
    community_spread: float = 1.0         # radius of community centroids
    intra_scale: float = 0.42             # size of each community blob
    intra_k: float = 0.55                 # spring length inside a community
    graph_width: float = 0.70             # fraction of the canvas for the graph

    # -- panels ------------------------------------------------------------
    show_node_type: bool = True
    show_edge_strength: bool = True
    show_communities: bool = True
    show_hubs: bool = True
    show_caption: bool = True
    hub_rows: int = 8
    panel_x: float = 0.715
    panel_w: float = 0.275
    panel_top: float = 0.960
    panel_bottom: float = 0.030
    panel_gap: float = 0.014
    row_pitch: float = 0.030

    # -- text --------------------------------------------------------------
    title: Optional[str] = None           # default: "<estimator> Covariance Network"
    subtitle: Optional[str] = None
    caption: Optional[str] = None
    dataset: Optional[str] = None
    top_pct: Optional[float] = None       # default: derived from the payload


class FigureUnavailable(RuntimeError):
    """Raised when the plotting extra is not installed."""


def rna_class(gene: str) -> str:
    g = str(gene).upper()
    if re.match(r"^(HSA-)?(MIR|LET-?7)", g):
        return "miRNA"
    if re.match(r"^(SNORD|SNORA|SCARNA)", g):
        return "snoRNA"
    return "protein_coding"


def _bucket(t: float, thresholds: Sequence[float]) -> int:
    for i, th in enumerate(reversed(list(thresholds))):
        if t >= th:
            return len(thresholds) - i
    return 0


def community_label(i: int, members: Sequence[str]) -> str:
    """Describe a module by what it contains — never by an assumed function."""
    order = ("protein_coding", "miRNA", "snoRNA")
    pretty = {"protein_coding": "protein-coding", "miRNA": "miRNA", "snoRNA": "snoRNA"}
    classes = [rna_class(m) for m in members]
    parts = [f"{classes.count(c)} {pretty[c]}" for c in order if classes.count(c)]
    return f"C{i + 1}: " + " · ".join(parts)


def _modular_layout(nx, np, G, comms, cfg: FigureConfig):
    """Communities as separated blobs: centroids on a circle, spring inside.

    Blob radius grows with community size, and the centroid circle grows with the
    number of communities, so many modules do not pile into each other.
    """
    pos: Dict[Any, Tuple[float, float]] = {}
    k = len(comms)
    biggest = max((len(c) for c in comms), default=1)
    # More communities need a bigger ring, else neighbouring blobs collide.
    R = cfg.community_spread * (1.0 if k <= 1 else max(1.0, 0.42 * k / 3.0))
    for i, comp in enumerate(comms):
        ang = 2 * np.pi * i / max(1, k)
        cx, cy = R * np.cos(ang), R * np.sin(ang)
        sub = G.subgraph(comp)
        if sub.number_of_nodes() == 1:
            pos[next(iter(comp))] = (cx, cy)
            continue
        p = nx.spring_layout(sub, seed=cfg.layout_seed, weight="w",
                             k=cfg.intra_k, iterations=cfg.layout_iterations)
        xs = [q[0] for q in p.values()]
        ys = [q[1] for q in p.values()]
        sx = (max(xs) - min(xs)) or 1.0
        sy = (max(ys) - min(ys)) or 1.0
        scale = cfg.intra_scale * (0.6 + 0.4 * (len(comp) / biggest) ** 0.5)
        for name, q in p.items():
            pos[name] = (cx + (q[0] / sx) * scale * 2, cy + (q[1] / sy) * scale * 2)
    return pos


def _declutter_labels(fig, annos, anchors_disp, iters: int) -> None:
    """Push overlapping labels apart so dense clusters stay readable.

    Measures each label once, then separates the boxes analytically in display
    space and writes the resulting offsets back — rather than re-drawing the
    canvas every iteration, which is far too slow for a few hundred labels.
    """
    if not annos:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(r) for a in annos]
    w = [b.width for b in boxes]
    h = [b.height for b in boxes]
    # Current label centres, in display px.
    cx = [(b.x0 + b.x1) / 2 for b in boxes]
    cy = [(b.y0 + b.y1) / 2 for b in boxes]
    ox = [c - a[0] for c, a in zip(cx, anchors_disp)]
    oy = [c - a[1] for c, a in zip(cy, anchors_disp)]

    n = len(annos)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[j] - cx[i]
                dy = cy[j] - cy[i]
                ox_need = (w[i] + w[j]) / 2 + 1.0
                oy_need = (h[i] + h[j]) / 2 + 1.0
                if abs(dx) >= ox_need or abs(dy) >= oy_need:
                    continue                      # not overlapping
                # Resolve along the axis needing the smaller push.
                push_x = ox_need - abs(dx)
                push_y = oy_need - abs(dy)
                if push_y <= push_x:
                    s = push_y / 2 + 0.4
                    sgn = 1.0 if dy >= 0 else -1.0
                    cy[i] -= sgn * s; cy[j] += sgn * s
                else:
                    s = push_x / 2 + 0.4
                    sgn = 1.0 if dx >= 0 else -1.0
                    cx[i] -= sgn * s; cx[j] += sgn * s
                moved = True
        if not moved:
            break
    # Keep labels tethered near their node so they stay identifiable.
    for i, a in enumerate(annos):
        dx = cx[i] - anchors_disp[i][0]
        dy = cy[i] - anchors_disp[i][1]
        lim = 46.0
        d = (dx * dx + dy * dy) ** 0.5
        if d > lim:
            dx, dy = dx * lim / d, dy * lim / d
        # display px -> points
        f = 72.0 / fig.dpi
        a.xyann = (dx * f, dy * f)


def render_network(
    payload: Dict[str, Any],
    *,
    fmt: str = "pdf",
    config: Optional[FigureConfig] = None,
    **overrides: Any,
) -> bytes:
    """Render the AD covariance network for *payload*; return the file bytes.

    *config* / *overrides* accept any :class:`FigureConfig` field.  ``top_pct``
    defaults to the edge budget implied by the payload itself (edges / all gene
    pairs), so the figure always describes what it actually shows.
    """
    cfg = config or FigureConfig()
    if overrides:
        unknown = set(overrides) - set(cfg.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown figure option(s): {sorted(unknown)}")
        cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

    try:
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

    p = payload.get("n_genes") or len(payload.get("genes") or [])
    top_pct = cfg.top_pct
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
    pos = _modular_layout(nx, np, G, comms, cfg)

    def nsize(n) -> float:
        return cfg.node_size_min + cfg.node_size_span * (deg[n] / maxdeg) ** cfg.node_size_gamma

    fig = Figure(figsize=cfg.figsize, dpi=cfg.dpi)
    FigureCanvasAgg(fig)                     # no pyplot: thread-safe
    fig.patch.set_facecolor(cfg.facecolor)
    ax = fig.add_axes([0.0, 0.0, cfg.graph_width, 0.905])
    ax.axis("off")

    for u, v, a in sorted(G.edges(data=True), key=lambda t: abs(t[2]["cov"])):
        b = _bucket(abs(a["cov"]) / maxabs, cfg.edge_thresholds)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=cfg.edge_ramp[b],
                linewidth=cfg.edge_widths[b], alpha=cfg.edge_alphas[b], zorder=1,
                solid_capstyle="round")

    for cls in ("protein_coding", "miRNA", "snoRNA"):
        ns = [n for n in G.nodes if rna_class(n) == cls]
        if not ns:
            continue
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns],
                   s=[nsize(n) for n in ns],
                   c=[cfg.community_colors[comm_of[n] % len(cfg.community_colors)] for n in ns],
                   marker=MARKER[cls], edgecolors=cfg.node_edge_color, linewidths=1.0,
                   zorder=3)

    # Fit the graph before measuring labels, so display coords are final.
    xs = [q[0] for q in pos.values()]
    ys = [q[1] for q in pos.values()]
    mx = (max(xs) - min(xs)) * 0.10 + 0.25
    my = (max(ys) - min(ys)) * 0.10 + 0.25
    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)

    hubs = sorted(deg, key=lambda n: -deg[n])
    hub_set = set(hubs[: max(0, cfg.hub_label_count)])
    annos, anchors = [], []
    if cfg.label_nodes:
        targets = hubs if cfg.max_labels is None else hubs[: cfg.max_labels]
        for n in targets:
            big = n in hub_set
            a = ax.annotate(
                str(n), pos[n], fontsize=cfg.hub_label_size if big else cfg.label_size,
                ha="center", va="center", zorder=4,
                fontweight="bold" if big else "normal",
                color=cfg.ink if big else "#33404f",
                xytext=(0, 9 + 0.010 * nsize(n) ** 0.5), textcoords="offset points",
                path_effects=[pe.withStroke(linewidth=2.6 if big else 2.0,
                                            foreground=cfg.facecolor)])
            annos.append(a)
            anchors.append(ax.transData.transform(pos[n]))
        if cfg.declutter:
            _declutter_labels(fig, annos, anchors, cfg.declutter_iters)

    # ---- titles ------------------------------------------------------------
    method = payload.get("recommended", "AD")
    nice = {"ad_target_ridge": "AD-Ridge", "ad_target_optimal": "AD-Optimal", "ad_target_lw": "AD-Ledoit-Wolf",
            "ad_target_oas": "AD-OAS", "ad_ridge": "AD-Ridge (hard projection)",
            "ad_linear_lw": "AD-Ledoit-Wolf (hard projection)",
            "ad_oas": "AD-OAS (hard projection)", "oas": "OAS", "lw": "Ledoit-Wolf",
            "ad_lasso": "AD-LASSO", "ad_elastic_net": "AD-Elastic-Net"}.get(method, method)
    title = cfg.title or f"{nice} Covariance Network"
    t = fig.text(0.012, 0.965, title, fontsize=21, fontweight="bold",
                 color=cfg.ink, va="top")
    fig.canvas.draw()
    tw = t.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()).width
    if top_pct:
        fig.text(0.012 + tw + 0.010, 0.9585,
                 f"(Top {top_pct:.3g}% Strongest Connections)",
                 fontsize=13.5, color=cfg.muted, va="top")
    src = payload.get("source") or {}
    ds = cfg.dataset or src.get("accession") or ("+".join(src.get("accessions", []))
                                                 or src.get("kind", "analysis"))
    n_samp = (payload.get("combined") or {}).get("n_samples_total") or payload.get("n_samples")
    tail = f"({n_samp} samples, {p} genes analysed)" if n_samp else f"({p} genes analysed)"
    bits = [b for b in [ds, cfg.subtitle] if b]
    fig.text(0.012, 0.928, " — ".join(bits) + " " + tail, fontsize=11.5,
             color=cfg.muted, style="italic", va="top")

    # ---- right-hand panel stack (fit-aware) --------------------------------
    # Panels are measured, then trimmed to fit the column. The previous version
    # used fixed arithmetic and silently ran off the page once a result had
    # enough communities.
    lx, lw = cfg.panel_x, cfg.panel_w
    fx = fig.transFigure

    def frac_h(points: float) -> float:
        """Points -> figure-height fraction. Fonts are in points, so any box
        sized in fractions must convert, or it overflows once figsize changes."""
        return points / 72.0 / cfg.figsize[1]

    HEADER_H, PAD_H = frac_h(21), frac_h(15)
    pitch = max(cfg.row_pitch, frac_h(11.5))

    cap_text = cfg.caption or (
        f"Edges represent covariance estimated by {nice}. Edge width and colour "
        "intensity are proportional to |covariance|; node size is proportional to "
        "degree."
        + (f" Only the strongest {top_pct:.3g}% of gene pairs are shown." if top_pct else "")
    )
    def wrap_chars(fontsize: float, pad_in: float = 0.10, bold: bool = False) -> int:
        """How many characters fit across the panel at *fontsize* points.

        Bold glyphs are appreciably wider, so headers need a larger per-character
        allowance or they run past the panel edge on a narrow canvas.
        """
        usable_pt = max(0.1, (lw * cfg.figsize[0] - pad_in * 2)) * 72.0
        return max(8, int(usable_pt / (fontsize * (0.66 if bold else 0.52))))

    cap_lines = textwrap.wrap(cap_text, width=wrap_chars(8.4)) if cfg.show_caption else []

    n_comm_rows = len(comms) if cfg.show_communities else 0
    n_hub_rows = min(cfg.hub_rows, len(hubs)) if cfg.show_hubs else 0

    def totals_lines(trimmed: int) -> List[str]:
        """The summary under the community list, wrapped to the panel width.

        Wrapped rather than emitted as one string: a long single line ran past
        the panel's right edge and was clipped by the canvas.
        """
        txt = (f"Total nodes = {G.number_of_nodes()}  ·  "
               f"Edges shown = {G.number_of_edges()}"
               + (f" (top {top_pct:.3g}%)" if top_pct else ""))
        if trimmed:
            txt += f"  ·  +{trimmed} more communities"
        return textwrap.wrap(txt, width=wrap_chars(8.6)) or [txt]

    def stack_height(nc: int, nh: int) -> float:
        h = 0.0
        if cfg.show_node_type or cfg.show_edge_strength:
            rows = max(3 if cfg.show_node_type else 0, 4 if cfg.show_edge_strength else 0)
            h += HEADER_H + rows * pitch + PAD_H + cfg.panel_gap
        if cfg.show_communities:
            nt = len(totals_lines(len(comms) - nc))
            h += HEADER_H + nc * pitch + PAD_H + frac_h(8.6 * 1.5) * nt + cfg.panel_gap
        if cfg.show_hubs:
            h += HEADER_H + (nh + 1) * pitch + PAD_H + cfg.panel_gap     # +header row
        if cfg.show_caption and cap_lines:
            h += frac_h(9) + frac_h(8.4 * 1.62) * len(cap_lines) + cfg.panel_gap
        return h

    avail = cfg.panel_top - cfg.panel_bottom
    trimmed_comm = trimmed_hub = 0
    # Trim the longest lists first, keeping the panels on the page.
    while stack_height(n_comm_rows, n_hub_rows) > avail and (n_comm_rows > 3 or n_hub_rows > 3):
        if n_comm_rows >= n_hub_rows and n_comm_rows > 3:
            n_comm_rows -= 1; trimmed_comm += 1
        elif n_hub_rows > 3:
            n_hub_rows -= 1; trimmed_hub += 1
    # Still too tall (tiny canvas): tighten the pitch rather than overflow.
    while stack_height(n_comm_rows, n_hub_rows) > avail and pitch > 0.016:
        pitch -= 0.001

    y = cfg.panel_top

    def panel(h: float) -> float:
        nonlocal y
        top = y
        fig.patches.append(FancyBboxPatch(
            (lx, top - h), lw, h, boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.9, edgecolor=cfg.rule, facecolor="white", transform=fx, zorder=5))
        y = top - h - cfg.panel_gap
        return top

    if cfg.show_node_type or cfg.show_edge_strength:
        rows = max(3 if cfg.show_node_type else 0, 4 if cfg.show_edge_strength else 0)
        h = HEADER_H + rows * pitch + PAD_H
        top = panel(h)
        if cfg.show_node_type:
            fig.text(lx + 0.012, top - 0.016, "Node type", fontsize=10,
                     fontweight="bold", color=cfg.ink, va="top", zorder=6)
            for i, cls in enumerate(("protein_coding", "miRNA", "snoRNA")):
                yy = top - HEADER_H - i * pitch - pitch * 0.35
                fig.text(lx + 0.030, yy, CLASS_LABEL[cls], fontsize=9, color=cfg.ink,
                         va="center", zorder=6)
                fig.lines.append(Line2D([lx + 0.019], [yy], marker=MARKER[cls],
                                        color="none", markerfacecolor="white",
                                        markeredgecolor=cfg.ink, markeredgewidth=1.1,
                                        markersize=8, transform=fx, zorder=6))
        if cfg.show_edge_strength:
            fig.text(lx + 0.148, top - 0.016, "Edge strength", fontsize=10,
                     fontweight="bold", color=cfg.ink, va="top", zorder=6)
            for i in range(4):
                yy = top - HEADER_H - i * pitch - pitch * 0.35
                fig.lines.append(Line2D([lx + 0.155, lx + 0.196], [yy, yy],
                                        color=cfg.edge_ramp[i],
                                        linewidth=cfg.edge_widths[i] + 0.9,
                                        alpha=max(cfg.edge_alphas[i], 0.75),
                                        transform=fx, zorder=6))
                fig.text(lx + 0.204, yy, STRENGTH_LABELS[i], fontsize=9, color=cfg.ink,
                         va="center", zorder=6)

    if cfg.show_communities:
        tlines = totals_lines(trimmed_comm)
        h = HEADER_H + n_comm_rows * pitch + PAD_H + frac_h(8.6 * 1.5) * len(tlines)
        top = panel(h)
        fig.text(lx + 0.012, top - 0.016, "Community (Louvain)", fontsize=10,
                 fontweight="bold", color=cfg.ink, va="top", zorder=6)
        for i in range(n_comm_rows):
            c = comms[i]
            yy = top - HEADER_H - i * pitch - pitch * 0.35
            fig.lines.append(Line2D([lx + 0.020], [yy], marker="o", color="none",
                                    markerfacecolor=cfg.community_colors[i % len(cfg.community_colors)],
                                    markeredgecolor="white", markersize=9,
                                    transform=fx, zorder=6))
            lab = community_label(i, sorted(c))
            room = wrap_chars(8.6) - 10          # leave space for the "n = ..." column
            if len(lab) > room:
                lab = lab[: max(4, room - 1)] + "…"
            fig.text(lx + 0.034, yy, lab, fontsize=8.6, color=cfg.ink,
                     va="center", zorder=6)
            fig.text(lx + lw - 0.014, yy, f"n = {len(c)}", fontsize=9, color=cfg.muted,
                     va="center", ha="right", zorder=6)
        fig.text(lx + 0.012, top - h + 0.018 * len(tlines) - 0.004,
                 "\n".join(tlines), fontsize=8.6, color=cfg.ink, va="top",
                 zorder=6, linespacing=1.45)

    if cfg.show_hubs and n_hub_rows:
        h = HEADER_H + (n_hub_rows + 1) * pitch + PAD_H
        top = panel(h)
        ttl = "Top hub nodes by degree"
        if trimmed_hub:
            ttl += f" (top {n_hub_rows})"
        room = wrap_chars(10, bold=True)
        if len(ttl) > room:
            ttl = ttl[: max(4, room - 1)] + "…"
        fig.text(lx + 0.012, top - 0.016, ttl, fontsize=10, fontweight="bold",
                 color=cfg.ink, va="top", zorder=6)
        hdr_y = top - HEADER_H - pitch * 0.35
        for lbl, dx, ha in (("Node", 0.034, "left"), ("Type", 0.135, "left"),
                            ("Degree", lw - 0.014, "right")):
            fig.text(lx + dx, hdr_y, lbl, fontsize=8.6, fontweight="bold",
                     color=cfg.muted, va="center", ha=ha, zorder=6)
        for i, n in enumerate(hubs[:n_hub_rows]):
            yy = hdr_y - (i + 1) * pitch
            cls = rna_class(n)
            # Same shape + community colour as the graph, by construction.
            fig.lines.append(Line2D([lx + 0.020], [yy], marker=MARKER[cls], color="none",
                                    markerfacecolor=cfg.community_colors[comm_of[n] % len(cfg.community_colors)],
                                    markeredgecolor="white", markersize=7.5,
                                    transform=fx, zorder=6))
            fig.text(lx + 0.034, yy, str(n), fontsize=8.6, color=cfg.ink, va="center", zorder=6)
            fig.text(lx + 0.135, yy, CLASS_LABEL[cls].replace(" gene", ""), fontsize=8.6,
                     color=cfg.ink, va="center", zorder=6)
            fig.text(lx + lw - 0.014, yy, str(deg[n]), fontsize=8.6, color=cfg.ink,
                     va="center", ha="right", zorder=6)

    if cfg.show_caption and cap_lines:
        h = frac_h(9) + frac_h(8.4 * 1.62) * len(cap_lines)
        top = y
        fig.patches.append(FancyBboxPatch(
            (lx, top - h), lw, h, boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.9, edgecolor="#bcd3ea", facecolor="#f4f9ff",
            transform=fx, zorder=5))
        fig.text(lx + 0.014, top - 0.012, "\n".join(cap_lines), fontsize=8.4,
                 color="#2c4a63", style="italic", va="top", zorder=6, linespacing=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, facecolor=cfg.facecolor, dpi=cfg.dpi)
    return buf.getvalue()
