#!/usr/bin/env python3
"""Render the covariance-network figure for the application note.

A thin CLI over :func:`adgencov.figure.render_network` — the same renderer the
service's "Download figure" button uses, so the manuscript figure and the one a
user downloads from the site are produced by identical code.

Any FigureConfig field can be set with --set NAME=VALUE, e.g.
    --set figsize=12,8 --set show_hubs=false --set community_colors=#111,#666

Usage:
    PYTHONPATH=python python3 scripts/paper_figure1.py \
        --result paper_out20/gse52778_result.json --out fig --top-pct 20
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields

from adgencov.figure import FigureConfig, render_network


def coerce(name: str, raw: str):
    """Parse a --set value against the FigureConfig field's declared type."""
    spec = {f.name: f for f in fields(FigureConfig)}[name]
    ann = str(spec.type)
    low = raw.strip().lower()
    if "bool" in ann:
        return low in ("1", "true", "yes", "on")
    if "Tuple[float, float]" in ann or name == "figsize":
        a, b = raw.split(",")
        return (float(a), float(b))
    if "Sequence[str]" in ann:
        return [v.strip() for v in raw.split(",") if v.strip()]
    if "Sequence[float]" in ann:
        return [float(v) for v in raw.split(",") if v.strip()]
    if "int" in ann and "Optional" not in ann:
        return int(raw)
    if "Optional[int]" in ann:
        return None if low in ("", "none") else int(raw)
    if "float" in ann and "Optional" not in ann:
        return float(raw)
    if "Optional[float]" in ann:
        return None if low in ("", "none") else float(raw)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="paper_out20/gse52778_result.json")
    ap.add_argument("--out", default="paper_out20/fig_network_ad_ridge")
    ap.add_argument("--top-pct", type=float, default=None,
                    help="Edge-budget label; derived from the payload if omitted.")
    ap.add_argument("--dataset", default="GSE52778")
    ap.add_argument("--subtitle", default="airway smooth muscle RNA-seq")
    ap.add_argument("--formats", default="pdf,jpeg")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="NAME=VALUE", help="Override any FigureConfig field.")
    ap.add_argument("--list-options", action="store_true",
                    help="Print every FigureConfig field and its default.")
    args = ap.parse_args()

    if args.list_options:
        for f in fields(FigureConfig):
            print(f"  {f.name:22s} {str(f.type).replace('typing.',''):28s} = {getattr(FigureConfig(), f.name)!r}")
        return

    valid = {f.name for f in fields(FigureConfig)}
    cfg_kw = {}
    for item in args.sets:
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in valid:
            raise SystemExit(f"unknown option {k!r}; --list-options shows them all")
        cfg_kw[k] = coerce(k, v)

    cfg = FigureConfig(top_pct=args.top_pct, dataset=args.dataset,
                       subtitle=args.subtitle, **cfg_kw)
    payload = json.load(open(args.result))
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        data = render_network(payload, fmt=fmt, config=cfg)
        path = f"{args.out}.{fmt}"
        with open(path, "wb") as fh:
            fh.write(data)
        print("wrote", path, f"({len(data)} bytes)")


if __name__ == "__main__":
    main()
