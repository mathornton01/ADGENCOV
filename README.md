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

## Citing

A Zenodo DOI will be minted on first tagged release. See `CITATION.cff`.

## License

MIT — see [LICENSE](LICENSE).

## Authors

Thornton & Thornton.
