# ADGENCOV

[![CI](https://github.com/mathornton01/ADGENCOV/actions/workflows/ci.yml/badge.svg)](https://github.com/mathornton01/ADGENCOV/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21206068.svg)](https://doi.org/10.5281/zenodo.21206068)

Algebraic-Diversity Genetic Covariance estimation for transcriptomics.

`adgencov` estimates gene-gene covariance networks from expression matrices
using **Algebraic Diversity (AD)**: a symmetry-group (Reynolds/commutant)
projection that regularizes the sample covariance toward a biologically
motivated invariant structure, either as a hard projection or as a shrinkage
target, combined with the classical shrinkage estimator family. The recommended
estimator is chosen automatically by cross-validated likelihood, and the
strongest covarying gene pairs are reported as a network.

The project ships four ways to run the same numerical core:

- a **C++17 command-line tool** (`adgencov`) for batch analysis,
- a **Python API** (pybind11) that exposes the whole core to NumPy,
- a **FastAPI HTTP service** with asynchronous jobs, and
- a **single-page web dashboard** served by that service, with a live
  covariance heatmap and force-directed network view.

## What it does

1. **Load and preprocess** an expression/count matrix: collapse duplicate gene
   symbols, drop low-abundance genes, optional `log2(x+1)`, select the most
   variable genes, and z-score.
2. **Build a symmetry partition** over genes (by family, chromosome, pathway, or
   data-driven correlation blocks) that defines the AD projection.
3. **Rank an estimator grid** by cross-validated Gaussian negative
   log-likelihood (exact leave-one-out, k-fold, or one-pass Extended BIC) and
   recommend the winner.
4. **Extract the covariance network**: the strongest covarying gene pairs, plus
   the full covariance matrix for the heatmap.

## Highlights

- High-performance C++17 numerical core (`libadgencov`) on Eigen, with the heavy
  routines releasing the GIL so the Python layer parallelizes across cores.
- Estimator family with an AD variant of each: sample, ridge, LASSO,
  elastic-net, Ledoit-Wolf, and OAS, plus AD-target shrinkage mixtures.
- Automatic model selection: exact leave-one-out CV, k-fold CV, or Extended BIC.
- Symmetry partitions from gene families, chromosomes, Reactome/GO/custom group
  maps, or data-driven correlation blocks and hierarchical wreaths.
- GEO ingestion by accession, including a supplementary-file fallback for
  RNA-seq count/FPKM/TPM matrices, with local caching.
- Optional enrichment against [STRING](https://string-db.org) and gene-symbol /
  protein-id resolution via mygene.info and UniProt.
- Every Python binding is parity-tested to ~1e-9 against the reference research
  prototype; CI runs the full suite on Linux, macOS, and Windows.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Eigen, Catch2, and pybind11 are fetched automatically. Relevant CMake options:

| Option | Default | Effect |
| --- | --- | --- |
| `ADGENCOV_BUILD_CLI` | ON | build the `adgencov` command-line tool |
| `ADGENCOV_BUILD_TESTS` | ON | build the Catch2 + pytest suite |
| `ADGENCOV_BUILD_PYTHON` | OFF | build the pybind11 module into `python/adgencov/` |
| `ADGENCOV_USE_LAPACK` | OFF | link LAPACK for the eigen-solver path |

## Command-line usage

The build produces the `adgencov` binary. A minimal run:

```sh
adgencov --expression counts.tsv --sample-regex '_LL[0-9]+' \
         --group gene_family --n-genes 500 --outdir results
```

It reads an expression/count matrix (delimiter auto-detected from the header),
preprocesses it, builds a symmetry partition, recommends an estimator by
leave-one-out NLL, and writes five files to `--outdir`:

- `gene_groups.csv`: the gene → group partition used for the AD projection
- `estimator_recommendations.csv`: every candidate ranked by LOO-NLL
- `best_covariance.csv`: the recommended covariance matrix (gene-labelled)
- `top_edges.csv`: the strongest covarying gene pairs
- `report.md`: a short run summary

Partition options for `--group`: `none`, `gene_family`, `chromosome` (needs
`--annotation`, columns `gene,chromosome`), `reactome` / `go_process` /
`custom_group_map` (need `--group-map`, columns `gene,group`), and the
clustering-based `correlation_blocks` / `hierarchical_wreath` (`--n-blocks`).
Run `adgencov --help` for the full option list and defaults (the CLI defaults to
`--n-genes 500` and leave-one-out selection).

Note: the CLI selects by leave-one-out NLL only. The k-fold and Extended BIC
criteria are exposed through the Python API and the service, not the CLI.

## Python API

The C++ core is exposed to Python via pybind11 (Eigen matrices convert
transparently to and from NumPy). Build the module, then use the wrapper:

```sh
cmake -S . -B build -DADGENCOV_BUILD_PYTHON=ON
cmake --build build -j          # drops adgencov/_core*.so into python/adgencov/
```

```python
import numpy as np, adgencov            # PYTHONPATH=python, or pip install -e python

X = np.random.default_rng(0).standard_normal((20, 8))   # samples x genes
labels = [0, 0, 1, 1, 1, 2, 2, 2]                        # group id per gene
result = adgencov.analyze(X, labels, criterion="loo")    # or "kfold" / "ebic"
print(result.best.spec.method, result.best.loo_nll)
print(result.to_dict())                                  # JSON-ready for the API
```

`analyze(...)` scores the estimator grid concurrently across cores (the C++
calls release the GIL) and returns an `AnalysisResult` whose `to_dict()` carries
the ranking, top covariance edges, gene blocks, and (for small gene sets, up to
600 genes) the full covariance matrix for the heatmap. Set
`ADGENCOV_MAX_WORKERS` to cap the pool.

Every bound function is parity-tested bit-for-bit (~1e-9) against the reference
prototype (`tests/test_bindings.py`, run by `ctest` as `python_bindings_parity`).
See `python/README.md` for details.

## GEO ingestion

`adgencov.geo` pulls a public expression series from NCBI GEO by accession and
runs it through the recommender. The series-matrix parser is pure pandas (no
`GEOparse` and no network needed to *parse*), so ingestion is fully
offline-testable; downloads are cached under `~/.cache/adgencov/geo` (override
with `$ADGENCOV_CACHE`).

```python
from adgencov import geo

result = geo.analyze_series("GSE52778", group="gene_family")   # download + analyze
print(result.best.spec.method, result.to_dict())

# Or parse a local series matrix (offline) and inspect before analyzing:
series = geo.read_series_matrix("GSE52778_series_matrix.txt.gz")
print(series.n_genes, series.n_samples, series.platform)
result = geo.analyze_series(series, n_genes=500)
```

When a series matrix carries only probe ids, they can be mapped to gene symbols
with `geo.map_probes_to_genes(series, mapping)`. For count/FPKM/TPM data that
lives only in the supplementary files, the loader can rank and pull a
matrix-shaped supplementary file (including `.tar` / `_RAW.tar` archives),
content-validate it, and drop annotation or differential-expression columns.
`GEOparse` is an optional extra (`pip install 'adgencov[geo]'`) for richer
platform annotation. The GEO to recommender path is verified end-to-end in
`tests/test_geo.py` (`ctest` target `geo_ingestion`) to 1e-9 against the
pipeline golden.

## HTTP service

`adgencov.api` is a FastAPI backend that turns the core, GEO ingestion, and the
database lookups into a web service. It also serves the dashboard. Analyses run
as asynchronous jobs and return the same `to_dict()` JSON as the Python API.

```sh
pip install 'adgencov[api]'                 # fastapi, uvicorn, pydantic, multipart, httpx
PYTHONPATH=python uvicorn adgencov.api:app  # dashboard at /, interactive docs at /docs
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness, version, active-job count |
| POST | `/analyze/upload` | multipart matrix + params → `202 {job}` |
| POST | `/analyze/geo` | JSON `{accession, ...params}` → `202 {job}` |
| GET | `/search/geo` | free-text GEO series search (NCBI E-utilities) |
| POST | `/translate/proteins` | protein/gene ids → protein names (UniProt) |
| POST | `/translate/symbols` | gene ids → official HGNC symbols (mygene.info) |
| GET | `/interactions` | STRING-db interaction partners for one or two genes |
| GET | `/jobs` | list jobs (newest first) |
| GET | `/jobs/{id}` | job detail; `result` present once `state=succeeded` |
| DELETE | `/jobs/{id}` | forget/cancel a job |
| GET | `/` | the single-page dashboard (static assets) |

```python
import time, httpx

c = httpx.Client(base_url="http://localhost:8000")
job = c.post("/analyze/geo", json={"accession": "GSE52778", "group": "gene_family"}).json()
while (d := c.get(f"/jobs/{job['id']}").json())["state"] not in ("succeeded", "failed"):
    time.sleep(1.0)
print(d["result"]["recommended"], d["result"]["ranking"][0]["loo_nll"])
```

Numerics run on a background thread pool (the C++ core releases the GIL), so the
API stays responsive under load. Uploads are capped at 50 MB. CORS is
preconfigured for `thorntonstatistical.com` and localhost dev servers. The
service is driven end-to-end over HTTP in `tests/test_api.py` (`ctest` target
`api_service`), and the dashboard in `tests/test_dashboard.py`
(`dashboard_service`), both fully offline.

The API and dashboard default to `n_genes = 150` for a fast interactive run; the
CLI and Python entry points default to 500. The dashboard exposes the k-fold and
Extended BIC selection criteria in addition to leave-one-out.

## Web dashboard

The dashboard (`python/adgencov/api/static/`) is a zero-build single-page client
served directly by the API. Enter a GEO accession (with keyword search) or
upload a matrix, watch the job progress, and explore the result: the estimator
ranking table, the gene-block view, a covariance heatmap (click a cell to look
up the two genes' STRING interactions), and a force-directed covariance network
with Louvain communities, RNA-class node shapes, and hub / top-pair panels. Gene
ids are relabelled to current symbols asynchronously via `/translate/symbols`.

## Repository layout

```
include/adgencov/   C++ public headers (projection, shrink, select, groups,
                    clustering, io, preprocess)
src/adgencov/       C++ implementations
cli/                the adgencov command-line tool (main.cpp)
bindings/           pybind11 module (adgencov._core)
python/adgencov/    Python package: analyze(), geo, bioquery, stringdb, biocache
python/adgencov/api/  FastAPI service + static dashboard
tests/              Catch2 (C++) + pytest (Python) suites and golden fixtures
cmake/              dependency fetching
```

The numerical modules: `projection` (the Reynolds/commutant projection and
symmetry builders), `shrink` (the estimator family and SPD repair), `select`
(dispatch, the CV / EBIC scoring rules, the candidate grid, and the
recommender), `groups` (symmetry partitions and factorization), `clustering`
(average-linkage agglomerative clustering for the data-driven partitions), `io`
(delimiter-sniffing table and matrix I/O), and `preprocess` (filtering,
transform, variable-gene selection, z-scoring).

## Testing

C++ tests run under Catch2 (`ctest` target `adgencov_tests`); Python tests run
under pytest and are registered individually with `ctest`
(`python_bindings_parity`, `geo_ingestion`, `api_service`, `dashboard_service`).
Golden values are generated from the reference prototype (`tests/adtarget_ref.py`
via the `gen_golden*.py` scripts) into committed headers, and the live tests
compare against NumPy, scikit-learn, and the prototype to ~1e-9. CI
(`.github/workflows/ci.yml`) builds and runs the full suite on Linux, macOS, and
Windows. A separate benchmark workflow (`bench.yml`) compares the LOO, k-fold,
and EBIC selection criteria on demand.

## Deployment

The service is containerized (multi-stage `Dockerfile`, portable SIMD build) and
deployed on Railway (`railway.json`, healthcheck at `/health`); see `DEPLOY.md`
and `DEPLOY_NOTES.md`. Auto-deploy is wired to `main`.

## Roadmap

The numerical core (projection, estimators, model selection, clustering) is
complete and verified, and the product layers on top of it are in place:

- **A. Python bindings**: pybind11 module + parity tests. Done.
- **B. GEO ingestion**: accession → matrix → pipeline, with caching. Done
  (`adgencov.geo`).
- **C. FastAPI backend**: async jobs over upload/accession → JSON, with auto
  OpenAPI docs. Done (`adgencov.api`).
- **D. Web dashboard**: ranking table, covariance heatmap, network graph, block
  view. Done (`api/static/`).
- **E. Compare-to-databases**: STRING interaction lookup and symbol/protein
  resolution wired into the dashboard. Done (`stringdb.py`, `bioquery.py`).
- **F. Desktop app**: a native Qt GUI over the same core. Planned.
- **G. Deploy + CI**: cross-platform CI green; containerized service deployed on
  Railway. Done.

## Citing

If you use ADGENCOV, please cite it via `CITATION.cff` or the archived release:
DOI [10.5281/zenodo.21206068](https://doi.org/10.5281/zenodo.21206068) (the
concept DOI `10.5281/zenodo.21206067` always resolves to the latest version).

## License

Modified MIT (Non-Commercial). The software is free to use, modify, and
redistribute for non-commercial purposes; commercial use requires a separate
written license from the authors. See [LICENSE](LICENSE) for the exact terms.

## Authors

Micah A. Thornton and Mitchell Thornton (Clearpoint Research LLC).
