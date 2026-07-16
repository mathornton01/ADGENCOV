"""Publication-ready exports of an analysis result.

One module renders the estimator-ranking table, the gene blocks, and the
covariance edges as LaTeX / CSV.  Both the manuscript scripts and the HTTP
service import it, so a table in the paper and a table downloaded from the tool
are produced by the same code and cannot drift apart — the drift between a
hand-transcribed table and the software is exactly what this avoids.

Everything takes the plain ``AnalysisResult.to_dict()`` payload (or a
``compare_series`` payload), so exports work for any job the service ran.
"""
from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple

# How the code's method names appear in the manuscript.  Equation (2) of the
# application note defines the AD family as the convex combination
# ``(1-lam) S + lam P_G(S)`` followed by ridge/LW/OAS -- that is the
# ``ad_target_*`` family.  The ``ad_*`` family is the lam=1 hard-projection
# special case and is labelled as such rather than as plain "AD-x".
METHOD_LABELS: Dict[str, str] = {
    "ad_target_ridge": "AD-Ridge (Eq. 2)",
    "ad_target_lw": "AD-Ledoit--Wolf (Eq. 2)",
    "ad_target_oas": "AD-OAS (Eq. 2)",
    "ad_ridge": "AD-Ridge (hard projection)",
    "ad_linear_lw": "AD-Ledoit--Wolf (hard projection)",
    "ad_oas": "AD-OAS (hard projection)",
    "ad_lasso": "AD-LASSO (hard projection)",
    "ad_elastic_net": "AD-Elastic-Net (hard projection)",
    "ad_sample": "AD-sample (hard projection)",
    "oas": "ordinary OAS",
    "lw": "ordinary Ledoit--Wolf",
    "ledoit_wolf": "ordinary Ledoit--Wolf",
    "ridge": "ridge",
    "lasso": "LASSO",
    "elastic_net": "Elastic-Net",
    "sample": "sample covariance",
}

#: Criterion -> the column header to print for the score.  ``loo_nll`` carries
#: whichever criterion was used, so labelling it "LOO-NLL" unconditionally would
#: be wrong for an EBIC or k-fold run.
SCORE_LABEL = {
    "loo": "Mean LOO-NLL",
    "kfold": "Mean k-fold NLL",
    "ebic": "EBIC",
}


def _fmt_params(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={v:g}" for k, v in sorted(params.items()))


def method_label(method: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Manuscript-facing label for a method, with its hyper-parameters."""
    base = METHOD_LABELS.get(method, method)
    ps = _fmt_params(params or {})
    return f"{base} ({ps})" if ps else base


def ranking_rows(payload: Dict[str, Any], limit: Optional[int] = None
                 ) -> List[Tuple[int, str, str, float, float]]:
    """(rank, method, params, score, condition_number) for each candidate."""
    rows = []
    for i, r in enumerate(payload.get("ranking", []), start=1):
        rows.append((i, r["method"], _fmt_params(r.get("params") or {}),
                     float(r["loo_nll"]), float(r.get("condition_number", float("nan")))))
    return rows[:limit] if limit else rows


def _tex_escape(s: str) -> str:
    # Labels here are method names/params; only these need escaping in practice.
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def ranking_to_latex(payload: Dict[str, Any], *, limit: Optional[int] = None,
                     criterion: str = "loo", caption: Optional[str] = None,
                     label: str = "tab:nll") -> str:
    """The estimator-ranking table as a LaTeX ``table`` float."""
    score_hdr = SCORE_LABEL.get(criterion, "Score")
    out = io.StringIO()
    if caption:
        out.write("\\begin{table}[h]\n\\centering\n")
        out.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
    out.write("\\begin{tabular}{llrr}\n\\toprule\n")
    out.write(f"Rank & Method & {score_hdr} & Cond.~\\#\\\\\n\\midrule\n")
    for rank, method, params, score, cond in ranking_rows(payload, limit):
        lab = _tex_escape(method_label(method, None))
        if params:
            lab += f" ({_tex_escape(params)})"
        cond_s = "--" if cond != cond or cond == float("inf") else f"{cond:.1f}"
        out.write(f"{rank} & {lab} & {score:.3f} & {cond_s}\\\\\n")
    out.write("\\bottomrule\n\\end{tabular}\n")
    if caption:
        out.write("\\end{table}\n")
    return out.getvalue()


def ranking_to_csv(payload: Dict[str, Any], *, limit: Optional[int] = None,
                   criterion: str = "loo") -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["rank", "method", "label", "params", SCORE_LABEL.get(criterion, "score"),
                "condition_number"])
    for rank, method, params, score, cond in ranking_rows(payload, limit):
        w.writerow([rank, method, METHOD_LABELS.get(method, method), params,
                    f"{score:.6f}", "" if cond != cond else f"{cond:.6f}"])
    return out.getvalue()


def edges_to_csv(payload: Dict[str, Any]) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["gene_a", "gene_b", "covariance", "abs_covariance"])
    for e in sorted(payload.get("edges", []), key=lambda x: -abs(x["covariance"])):
        w.writerow([e["gene_a"], e["gene_b"], f"{e['covariance']:.6f}",
                    f"{e['abs_covariance']:.6f}"])
    return out.getvalue()


def blocks_to_csv(payload: Dict[str, Any]) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["gene", "block"])
    genes = payload.get("genes", [])
    labels = payload.get("labels", [])
    for g, b in zip(genes, labels):
        w.writerow([g, b])
    return out.getvalue()


def covariance_to_csv(payload: Dict[str, Any]) -> str:
    """The recommended estimator's covariance matrix, gene-labelled."""
    cov = payload.get("covariance")
    genes = payload.get("genes", [])
    if not cov:
        raise ValueError("this result carries no covariance matrix "
                         "(omitted above the payload size cap)")
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow([""] + list(genes))
    for g, row in zip(genes, cov):
        w.writerow([g] + [f"{v:.6f}" for v in row])
    return out.getvalue()


# ---------------------------------------------------------------------------
# multi-dataset compare
# ---------------------------------------------------------------------------
def compare_to_csv(payload: Dict[str, Any]) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["section", "a", "b", "value", "detail"])
    for d in payload.get("datasets", []):
        w.writerow(["dataset", d["accession"], "", d.get("recommended", ""),
                    f"score={d.get('loo_nll')}; samples={d.get('n_samples')}; "
                    f"edges={d.get('n_edges')}"])
    for p in payload.get("comparison", {}).get("pairs", []):
        w.writerow(["pair", p["a"], p["b"], f"jaccard={p['edge_jaccard']:.6f}",
                    f"shared_edges={p['shared_edges']}; sign_agreement={p['sign_agreement']}; "
                    f"same_recommendation={p['same_recommendation']}"])
    for e in payload.get("comparison", {}).get("recurrent_edges", []):
        w.writerow(["recurrent_edge", e["gene_a"], e["gene_b"], e["n_datasets"], ""])
    return out.getvalue()


def compare_to_latex(payload: Dict[str, Any], *, caption: Optional[str] = None,
                     label: str = "tab:compare") -> str:
    out = io.StringIO()
    if caption:
        out.write("\\begin{table}[h]\n\\centering\n")
        out.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
    out.write("\\begin{tabular}{llrr}\n\\toprule\n")
    out.write("Dataset & Recommended & Score & Edges\\\\\n\\midrule\n")
    for d in payload.get("datasets", []):
        score = d.get("loo_nll")
        out.write(f"{_tex_escape(d['accession'])} & "
                  f"{_tex_escape(method_label(d.get('recommended', '')))} & "
                  f"{score:.3f} & {d.get('n_edges', 0)}\\\\\n"
                  if score is not None else
                  f"{_tex_escape(d['accession'])} & -- & -- & --\\\\\n")
    out.write("\\bottomrule\n\\end{tabular}\n")
    if caption:
        out.write("\\end{table}\n")
    return out.getvalue()
