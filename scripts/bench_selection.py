#!/usr/bin/env python3
"""Sandbox bench-off of estimator-selection criteria for ADGENCOV.

Compares three criteria for choosing among the candidate_grid estimators:

  * LOO  - leave-one-out CV NLL (the incumbent criterion)
  * EBIC - Extended BIC (Foygel & Drton), Method 1: one full-sample penalized
           likelihood pass per candidate
  * KFOLD- k-fold CV NLL (k=5), Method 2: k refits per candidate

Ground truth: data are drawn from a known covariance in the p >> n regime.
Each criterion selects an estimator on a small TRAIN set; the selected
estimator is then scored by its mean Gaussian NLL on a large, independent
HELD-OUT TEST set (many samples), which is an unbiased Monte-Carlo estimate of
its true expected generalization loss.  The criterion whose selections
generalize best (lowest held-out test NLL), most often match the oracle grid
choice, and cost the least wall-clock, wins.

Two truth regimes are run:
  A) block-exchangeable (AD-symmetric) truth  - the AD projection is exactly
     correct, so heavy AD shrinkage should win;
  B) perturbed truth                          - block structure + a dense random
     perturbation, so the AD assumption is mildly misspecified.
"""
import os
import time
import numpy as np

from adgencov import _core, candidate_grid, gaussian_nll_one, estimate_covariance

RNG = np.random.default_rng(20260707)


def make_labels(p, n_blocks):
    base = np.repeat(np.arange(n_blocks), p // n_blocks)
    extra = np.arange(p - base.size) % n_blocks
    return np.concatenate([base, extra]).astype(int).tolist()


def block_exchangeable_cov(p, labels, within=0.6, between=0.15, diag=1.0):
    """A covariance constant on the block-exchangeable orbits (AD-symmetric)."""
    labels = np.asarray(labels)
    C = np.full((p, p), 0.0)
    same = labels[:, None] == labels[None, :]
    C[same] = between
    for b in np.unique(labels):
        idx = np.where(labels == b)[0]
        C[np.ix_(idx, idx)] = within
    np.fill_diagonal(C, diag)
    # nudge to SPD
    w = np.linalg.eigvalsh(C).min()
    if w <= 0:
        C += (1e-3 - w) * np.eye(p)
    return C


def perturb(C, scale, rng):
    A = rng.standard_normal(C.shape)
    S = scale * (A + A.T) / 2.0
    np.fill_diagonal(S, 0.0)
    Cp = C + S
    w = np.linalg.eigvalsh(Cp).min()
    if w <= 0:
        Cp += (1e-3 - w) * np.eye(C.shape[0])
    return Cp


def draw(n, cov, rng):
    L = np.linalg.cholesky(cov)
    return (rng.standard_normal((n, cov.shape[0])) @ L.T)


def standardize(X):
    mu = X.mean(0)
    sd = X.std(0, ddof=1)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def test_nll(spec, Xtr, labels, Xte):
    """Mean Gaussian NLL of `spec` (fit on Xtr) over the held-out Xte rows."""
    Sigma = np.asarray(estimate_covariance(Xtr, labels, spec), float)
    mu = Xtr.mean(0)
    return float(np.mean([gaussian_nll_one(Xte[i], mu, Sigma) for i in range(Xte.shape[0])]))


def score_grid(criterion, Xtr, labels, grid, gamma=0.5, k=5):
    scores = []
    for sp in grid:
        try:
            if criterion == "loo":
                s = _core.loo_nll(Xtr, labels, sp)
            elif criterion == "ebic":
                s = _core.ebic_score(Xtr, labels, sp, gamma)
            elif criterion == "kfold":
                s = _core.kfold_nll(Xtr, labels, sp, k)
            else:
                raise ValueError(criterion)
        except Exception:
            s = np.inf
        scores.append(s)
    return np.asarray(scores)


def run_regime(name, cov, p, labels, n_train, n_test, reps):
    grid = candidate_grid(p, n_train)
    crits = ["loo", "ebic", "kfold"]
    gen = {c: [] for c in crits}       # held-out NLL of each criterion's pick
    oracle_gen = []                    # held-out NLL of the true best grid member
    match = {c: 0 for c in crits}      # times criterion picked the oracle estimator
    picks = {c: [] for c in crits}
    timing = {c: 0.0 for c in crits}

    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        Xtr = standardize(draw(n_train, cov, rng))
        Xte = standardize(draw(n_test, cov, rng))

        # oracle: grid member with the lowest held-out test NLL this replicate
        te = np.array([test_nll(sp, Xtr, labels, Xte) for sp in grid])
        oracle_idx = int(np.argmin(te))
        oracle_gen.append(te[oracle_idx])

        for c in crits:
            t0 = time.perf_counter()
            sc = score_grid(c, Xtr, labels, grid)
            timing[c] += time.perf_counter() - t0
            pick = int(np.argmin(sc))
            picks[c].append(grid[pick].method)
            gen[c].append(te[pick])
            if pick == oracle_idx:
                match[c] += 1

    print(f"\n=== Regime {name}  (p={p}, n_train={n_train}, n_test={n_test}, reps={reps}) ===")
    o = np.mean(oracle_gen)
    print(f"  oracle held-out NLL (best possible): {o:.4f}")
    print(f"  {'criterion':8s}  {'meanTestNLL':>11s}  {'regret':>8s}  {'oracle%':>7s}  {'sec/rep':>8s}")
    summary = {}
    for c in crits:
        m = np.mean(gen[c])
        regret = m - o
        omatch = 100.0 * match[c] / reps
        spr = timing[c] / reps
        summary[c] = (m, regret, omatch, spr)
        print(f"  {c:8s}  {m:11.4f}  {regret:8.4f}  {omatch:6.1f}%  {spr:8.4f}")
    return summary


def main():
    p, n_blocks = 60, 4
    labels = make_labels(p, n_blocks)
    n_train, n_test, reps = 20, 600, 30

    covA = block_exchangeable_cov(p, labels)
    covB = perturb(covA, scale=0.05, rng=RNG)

    sumA = run_regime("A: block-exchangeable (AD exact)", covA, p, labels, n_train, n_test, reps)
    sumB = run_regime("B: perturbed (AD misspecified)", covB, p, labels, n_train, n_test, reps)

    # Regime C: genomics scale (GSE52778-like: p >> n, tiny n). This is the
    # regime dad cares about -- LOO does n full refits per grid point, so its
    # wall-clock explodes while EBIC stays one pass. Does its accuracy edge hold?
    pC, nbC = 300, 6
    labelsC = make_labels(pC, nbC)
    covC0 = block_exchangeable_cov(pC, labelsC)
    covC = perturb(covC0, scale=0.03, rng=RNG)
    # Reps for the (expensive) genomics-scale regime are configurable so the
    # CI bench workflow can trade wall-clock for tighter numbers via its `reps`
    # input (BENCH_REPS_C). Defaults to 12 when run locally.
    reps_c = int(os.environ.get("BENCH_REPS_C") or 12)
    sumC = run_regime("C: genomics scale p>>n (perturbed)", covC, pC, labelsC,
                      n_train=16, n_test=400, reps=reps_c)

    print("\n=== VERDICT ===")
    crits = ["loo", "ebic", "kfold"]
    # combined mean regret across all three regimes (lower is better)
    combined = {c: (sumA[c][1] + sumB[c][1] + sumC[c][1]) / 3 for c in crits}
    speed = {c: (sumA[c][3] + sumB[c][3] + sumC[c][3]) / 3 for c in crits}
    order = sorted(crits, key=lambda c: combined[c])
    for c in order:
        print(f"  {c:8s}  mean regret={combined[c]:.4f}   sec/rep={speed[c]:.4f}")
    win = order[0]
    print(f"\n  WINNER (lowest generalization regret): {win.upper()}")
    if win != "loo":
        print(f"  Speedup vs LOO: {speed['loo']/speed[win]:.1f}x")
    # Scale story: how LOO's cost blows up at genomics p vs the fast methods
    print("\n  --- genomics-scale (regime C) wall-clock per criterion ---")
    for c in crits:
        print(f"    {c:8s}  {sumC[c][3]:8.4f} sec/rep   regret={sumC[c][1]:.4f}")
    print(f"    LOO is {sumC['loo'][3]/sumC['ebic'][3]:.1f}x slower than EBIC, "
          f"{sumC['loo'][3]/sumC['kfold'][3]:.1f}x slower than KFOLD at p={pC}")


if __name__ == "__main__":
    main()
