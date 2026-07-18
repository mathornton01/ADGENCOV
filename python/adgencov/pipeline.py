"""Shared analysis orchestration: estimator selection + automatic grouping.

Both service entry points (uploaded matrix, GEO accession) share two concerns
that do not belong in the numerical core:

* :func:`choose_group` — when the user asks for ``group="auto"``, run the
  recommender under several built-in symmetry structures and keep the one whose
  best estimator has the lowest cross-validated score.  This answers "try various
  built-in structures and pick the best" without the user guessing a grouping.
* :func:`grouping_meta` — a small JSON block describing which grouping was used
  (and, for auto, the ranked candidates) so the dashboard and exports can show it.

The caller supplies a ``run_one(group, n_blocks, progress) -> AnalysisResult``
closure, so this module never needs to know whether the data came from an upload
or from GEO.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

# Built-in structures tried by group="auto".  Only ones needing no extra files:
# chromosome needs an annotation table and the pathway/custom groups need a group
# map, so they are offered explicitly rather than swept blindly.  correlation
# blocks are swept at a few resolutions since the right block count is unknown.
AUTO_GROUP_PLAN: Tuple[Tuple[str, Optional[int]], ...] = (
    ("none", None),
    ("gene_family", None),
    ("correlation_blocks", 2),
    ("correlation_blocks", 4),
    ("correlation_blocks", 8),
)

AnalysisResultLike = Any
RunOne = Callable[[str, int, Optional[Callable[[float, str], None]]], AnalysisResultLike]


def _gid(group: str, n_blocks: int) -> str:
    return f"{group} (n_blocks={n_blocks})" if group == "correlation_blocks" else group


def _band(progress, lo: float, hi: float):
    if progress is None:
        return None

    def scaled(fraction: float, phase: str) -> None:
        progress(lo + (hi - lo) * fraction, phase)

    return scaled


def choose_group(
    run_one: RunOne,
    *,
    default_n_blocks: int = 4,
    plan: Tuple[Tuple[str, Optional[int]], ...] = AUTO_GROUP_PLAN,
    progress=None,
) -> Tuple[AnalysisResultLike, Dict[str, Any]]:
    """Run *run_one* under each planned grouping; return (best_result, meta).

    "Best" is the lowest ``best.loo_nll`` (whatever criterion the run used — all
    candidates share it, so the scores are comparable).  A grouping that fails to
    build or fit is recorded and skipped rather than aborting the whole run.
    """
    steps = [(g, nb if nb is not None else default_n_blocks) for g, nb in plan]
    tried: List[Dict[str, Any]] = []
    best: Optional[Tuple[str, float, AnalysisResultLike]] = None
    for i, (group, nb) in enumerate(steps):
        rep = _band(progress, i / len(steps), (i + 1) / len(steps))
        gid = _gid(group, nb)
        try:
            res = run_one(group, nb, rep)
            score = float(res.best.loo_nll)
        except Exception as exc:  # noqa: BLE001 - one bad grouping must not abort
            tried.append({"group": gid, "error": str(exc)[:140]})
            continue
        tried.append({"group": gid, "recommended": res.best.spec.method, "score": score})
        if best is None or score < best[1]:
            best = (gid, score, res)
    if best is None:
        raise RuntimeError(
            "automatic grouping produced no valid fit "
            + "; ".join(t.get("error", "") for t in tried if "error" in t)
        )
    meta = {
        "mode": "auto",
        "chosen": best[0],
        "candidates": sorted(
            (t for t in tried if "score" in t), key=lambda t: t["score"]
        ),
        "skipped": [t for t in tried if "error" in t],
    }
    return best[2], meta


def grouping_meta(group: str, n_blocks: int) -> Dict[str, Any]:
    """Metadata block for a fixed (non-auto) grouping."""
    return {"mode": "fixed", "chosen": _gid(group, n_blocks)}
