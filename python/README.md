# adgencov (Python)

Friendly Python API over the ADGENCOV C++ core, bound with pybind11.

The numerically heavy pipeline — the Reynolds/commutant projection, the
estimator family, leave-one-out model selection, and agglomerative
clustering — runs in vectorised Eigen/C++. This package is the language
boundary the higher product layers use: GEO ingestion, the FastAPI service,
and the web/desktop GUIs all drive the fast path through `import adgencov`.

## Building the extension

The compiled module `adgencov._core` is produced by the top-level CMake build,
which drops `_core*.so` directly into this package directory:

```bash
cmake -S .. -B ../build-py -DADGENCOV_BUILD_PYTHON=ON
cmake --build ../build-py -j
```

Then, from anywhere with this directory on `PYTHONPATH` (or after an editable
install):

```python
import numpy as np, adgencov

X = np.random.default_rng(0).standard_normal((20, 8))   # samples x genes
labels = [0, 0, 1, 1, 1, 2, 2, 2]                        # group per gene
result = adgencov.analyze(X, labels)
print(result.best.spec.method, result.best.loo_nll)
print(result.to_dict())                                  # JSON-ready for the API
```

## Parity

Every bound function is checked bit-for-bit (~1e-9) against the original
prototype in `tests/test_bindings.py`, which also runs as the `ctest` target
`python_bindings_parity`.

## Roadmap position

This is Phase A of the product roadmap (see the repository `README.md`). Later
phases add GEO ingestion (`adgencov[geo]`), the FastAPI backend
(`adgencov[api]`), and the web/desktop GUIs.
