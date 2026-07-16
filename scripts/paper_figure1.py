#!/usr/bin/env python3
"""Render the covariance-network figure for the application note.

A thin CLI over :func:`adgencov.figure.render_network` — the same renderer the
service's "Download figure" button uses, so the manuscript figure and the one a
user downloads from the site are produced by identical code.

Usage:
    PYTHONPATH=python python3 scripts/paper_figure1.py \
        --result paper_out20/gse52778_result.json \
        --out paper_out20/fig_network_ad_ridge --top-pct 20
"""
from __future__ import annotations

import argparse
import json

from adgencov.figure import render_network


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="paper_out20/gse52778_result.json")
    ap.add_argument("--out", default="paper_out20/fig_network_ad_ridge")
    ap.add_argument("--top-pct", type=float, default=None,
                    help="Label for the edge budget; derived from the payload if omitted.")
    ap.add_argument("--dataset", default="GSE52778")
    ap.add_argument("--subtitle", default="airway smooth muscle RNA-seq")
    ap.add_argument("--formats", default="pdf,jpeg")
    args = ap.parse_args()

    payload = json.load(open(args.result))
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        data = render_network(payload, fmt=fmt, top_pct=args.top_pct,
                              dataset=args.dataset, subtitle=args.subtitle)
        path = f"{args.out}.{fmt}"
        with open(path, "wb") as fh:
            fh.write(data)
        print("wrote", path, f"({len(data)} bytes)")


if __name__ == "__main__":
    main()
