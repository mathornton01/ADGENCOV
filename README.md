# ADGENCOV

[![CI](https://github.com/mathornton01/ADGENCOV/actions/workflows/ci.yml/badge.svg)](https://github.com/mathornton01/ADGENCOV/actions/workflows/ci.yml)

Algebraic-Diversity Genetic Covariance estimation for transcriptomics.

`adgencov` estimates gene–gene covariance and precision (partial-correlation)
networks from expression matrices, using **Algebraic Diversity (AD)** — a
symmetry-group Reynolds projection that regularizes the covariance toward a
biologically motivated invariant structure before/after classical shrinkage.

- High-performance C++17 numerical core (`libadgencov`) with a stable C ABI
- Shrinkage estimators: ridge, Ledoit–Wolf, OAS, LASSO, elastic-net, each with an AD variant
- LOO-NLL model selection to auto-recommend the best estimator
- Bioinformatics-flavored CLI: TSV/CSV/FPKM/TPM/`.mtx` I/O, GMT group maps
  (Reactome/GO/KEGG/MSigDB), GraphML/edge-list/community outputs for Cytoscape
- Python bindings (pybind11) that keep the original research prototype working
- Cross-platform: Linux, macOS, Windows — pip wheels, conda-forge, Homebrew

## Status

Early scaffold. See `docs/` for the user and technical manuals (in progress)
and the Oxford Bioinformatics Applications Note under `paper/`.

## Build (preview)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Command-line usage

Build enables the `adgencov` tool by default. A minimal run:

```sh
adgencov --expression counts.tsv --sample-regex '_LL[0-9]+' \
         --group gene_family --n-genes 500 --outdir results
```

Reads an expression/count matrix (delimiter auto-detected), preprocesses it
(duplicate-symbol collapse, low-expression filter, log2, variable-gene
selection, z-score), builds a symmetry partition, recommends an estimator by
leave-one-out NLL, and writes to `--outdir`:

- `gene_groups.csv` — the gene → group partition used for the AD projection
- `estimator_recommendations.csv` — every candidate ranked by LOO-NLL
- `best_covariance.csv` — the recommended covariance matrix (gene-labelled)
- `top_edges.csv` — the strongest covarying gene pairs
- `report.md` — a short run summary

Partitions available now: `none`, `gene_family`, `chromosome` (needs
`--annotation`), `reactome`/`go_process`/`custom_group_map` (need
`--group-map`, columns `gene,group`), and the clustering-based
`correlation_blocks` / `hierarchical_wreath` (`--n-blocks`). Network community
detection lands in a later release. Run `adgencov --help` for the full option
list.

## Python API

The full C++ core is exposed to Python via pybind11 (Eigen matrices convert
transparently to/from NumPy). Build the module and use the friendly wrapper:

```sh
cmake -S . -B build -DADGENCOV_BUILD_PYTHON=ON
cmake --build build -j          # drops adgencov/_core*.so into python/adgencov/
```

```python
import numpy as np, adgencov            # PYTHONPATH=python, or pip install -e python

X = np.random.default_rng(0).standard_normal((20, 8))   # samples x genes
labels = [0, 0, 1, 1, 1, 2, 2, 2]                        # group id per gene
result = adgencov.analyze(X, labels)
print(result.best.spec.method, result.best.loo_nll)
print(result.to_dict())                                  # JSON-ready for the API
```

Every bound function is parity-tested bit-for-bit (~1e-9) against the reference
prototype (`tests/test_bindings.py`, also run by `ctest` as
`python_bindings_parity`). See `python/README.md` for details.

## GEO ingestion

`adgencov.geo` pulls a public expression series straight from NCBI GEO by
accession and runs it through the recommender. The series-matrix parser is
pure-pandas (no `GEOparse`, no network needed to *parse*), so ingestion is
fully offline-testable; downloads are cached under `~/.cache/adgencov/geo`.

```python
from adgencov import geo

result = geo.analyze_series("GSE52778", group="gene_family")   # download + analyze
print(result.best.spec.method, result.to_dict())

# Or parse a local series matrix (offline) and inspect before analyzing:
series = geo.read_series_matrix("GSE52778_series_matrix.txt.gz")
print(series.n_genes, series.n_samples, series.platform)
result = geo.analyze_series(series, n_genes=500)
```

Rows keyed by platform probe ids can be mapped to gene symbols with
`geo.map_probes_to_genes(series, mapping)` before analysis. `GEOparse` is an
optional extra (`pip install 'adgencov[geo]'`) for richer platform annotation.
The GEO→recommender path is verified end-to-end in `tests/test_geo.py` (run by
`ctest` as `geo_ingestion`): on a fixture sharing the CLI golden's genes it
reproduces the reference recommendation to 1e-9.

## HTTP service

`adgencov.api` is a FastAPI backend that turns the core + GEO ingestion into a
web service — the server the web dashboard (Phase D) and hosted portal call.
Submit an uploaded matrix or a GEO accession; the work runs as an asynchronous
job and returns the same `to_dict()` JSON as the Python API.

```sh
pip install 'adgencov[api]'                 # fastapi, uvicorn, python-multipart, httpx
PYTHONPATH=python uvicorn adgencov.api:app  # interactive docs at /docs
```

```
GET    /health                 liveness + version + active-job count
POST   /analyze/upload         multipart matrix (TSV/CSV) + params  -> 202 {job}
POST   /analyze/geo            JSON {accession, ...params}          -> 202 {job}
GET    /jobs                   list jobs (newest first)
GET    /jobs/{id}              job detail; result present once state=succeeded
DELETE /jobs/{id}              forget/cancel a job
```

```python
import time, httpx

c = httpx.Client(base_url="http://localhost:8000")
job = c.post("/analyze/geo", json={"accession": "GSE52778", "group": "gene_family"}).json()
while (d := c.get(f"/jobs/{job['id']}").json())["state"] not in ("succeeded", "failed"):
    time.sleep(1.0)
print(d["result"]["recommended"], d["result"]["ranking"][0]["loo_nll"])
```

Numerics run on a background thread pool (the C++ core releases the GIL), so the
API stays responsive under load. CORS is preconfigured for
`thorntonstatistical.com` and localhost dev servers. The service is driven
end-to-end over HTTP in `tests/test_api.py` (run by `ctest` as `api_service`):
it proves the web boundary doesn't perturb the numerics and reproduces the 1e-9
pipeline golden through the GEO endpoint — fully offline.

## Roadmap — from library to product

The numerical core (projection, estimators, model selection, clustering) is
complete and verified. The product layers build on top of it:

- **A. Python bindings** — pybind11 module + parity tests. ✅ *done*
- **B. GEO ingestion** — pull transcriptomics series by accession → matrix →
  pipeline, with local caching. ✅ *done* (`adgencov.geo`)
- **C. FastAPI backend** — async jobs over upload/accession → JSON
  (recommendations, covariance, edges, blocks); auto OpenAPI docs. ✅ *done*
  (`adgencov.api`)
- **D. Web dashboard** — accession/upload → ranking table, covariance heatmap,
  network graph, block view.
- **E. Compare-to-databases** — cross-reference discovered gene blocks and
  covariance edges against the [STRING](https://string-db.org) protein–protein
  interaction database (functional enrichment + edge overlap).
- **F. Desktop app** — a native **Qt** GUI over the same service/core, for a
  standalone offline scientific tool.
- **G. Deploy + CI** — containerized service hosted on
  **thorntonstatistical.com**; cross-platform test matrix already green (CI). ✅
  *CI done*

## Citing

A Zenodo DOI will be minted on first tagged release. See `CITATION.cff`.

## License

MIT — see [LICENSE](LICENSE).

## Authors

Thornton & Thornton.
