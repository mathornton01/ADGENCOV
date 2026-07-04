"""Quick before/after benchmark for the estimator recommender.

Times the full candidate grid on a synthetic matrix shaped like GSE2034
(n samples x p genes). Run with the golden venv + built module:

    PYTHONPATH=python .goldenv/bin/python bench_speed.py [p]
"""
import sys
import time

import numpy as np

import adgencov
from adgencov import candidate_grid, estimate_covariance, loo_nll

n = 286
p = int(sys.argv[1]) if len(sys.argv) > 1 else 200

rng = np.random.default_rng(0)
# Two-block structured data so the AD (Reynolds) variants are exercised.
X = rng.standard_normal((n, p))
labels = [i % 4 for i in range(p)]

grid = candidate_grid(p, n)
print(f"n={n} p={p} candidates={len(grid)}")

# Warm up import/JIT of numpy paths.
_ = loo_nll(X, labels, grid[0])

t0 = time.perf_counter()
result = adgencov.analyze(X, labels)
t1 = time.perf_counter()
best = result.ranking[0]
print(f"analyze()  wall={t1 - t0:8.2f}s  best={best.spec.method:18s} loo={best.loo_nll:.6f}")
