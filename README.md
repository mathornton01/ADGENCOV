# ADGENCOV

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
`--annotation`), and `reactome`/`go_process`/`custom_group_map` (need
`--group-map`, columns `gene,group`). Clustering-based partitions
(`correlation_blocks`, `hierarchical_wreath`) and network community detection
land in a later release. Run `adgencov --help` for the full option list.

## Citing

A Zenodo DOI will be minted on first tagged release. See `CITATION.cff`.

## License

MIT — see [LICENSE](LICENSE).

## Authors

Thornton & Thornton.
