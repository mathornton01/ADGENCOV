# Derived analysis outputs

These are the outputs backing the ADGENCOV application note. Every file here is
regenerated end-to-end from the released software by a script in `scripts/`; none
of it is transcribed by hand. Raw GEO matrices are **not** vendored — the scripts
download and cache them on first run.

## `gse52778/` — main-text example (gene-family symmetry)

Produced by:

```sh
PYTHONPATH=python python3 scripts/paper_gse52778.py \
    --outdir data/gse52778 --group gene_family
PYTHONPATH=python python3 scripts/treatment_rewiring.py --outdir data/gse52778
PYTHONPATH=python python3 scripts/paper_figure1.py \
    --result data/gse52778/gse52778_result.json \
    --out data/gse52778/fig_network_gene_family --top-pct 1
```

| File | Contents |
| --- | --- |
| `gene_groups.csv` | gene → symmetry group partition used for the AD projection |
| `estimator_ranking.csv` | all 40 candidates ranked by exact leave-one-out NLL |
| `best_covariance.csv` | the recommended covariance matrix, gene-labelled |
| `top_edges.csv` | the strongest 1% of covariance edges (20 pairs) |
| `communities.csv` | Louvain community assignment over the retained edges |
| `treatment_rewiring.json` | treatment-stratified edge lists and sign changes |
| `gse52778_result.json` | the full analysis payload (also the figure input) |
| `table1.tex` | Table 1 of the note, as LaTeX |
| `report.md` | human-readable run summary |
| `fig_network_gene_family.{pdf,jpeg}` | Figure 1 of the note |

Headline result: AD-Ridge (Eq. 2, λ = 0.7) achieves a mean LOO-NLL of **80.452**
against **85.165** for the best ordinary estimator. Note that AD-OAS and
AD-Ledoit–Wolf do *not* beat their ordinary counterparts under this symmetry —
the recommender scores each candidate on held-out likelihood rather than
assuming the symmetry prior helps.

## `gse52778_corrblocks/` — supplementary comparison

The same pipeline under the data-driven four-block correlation surrogate
(`--group correlation_blocks --n-blocks 4`), which the Supplementary Information
compares against the gene-family symmetry. AD-Ridge still ranks first here, but
by only 0.167 NLL.

## `synthetic_probe/` — supplementary Table S1

Produced by `scripts/synthetic_probe.py`. Compares the best AD estimator against
the best ordinary shrinkage estimator across four generative regimes (matched
block, weak block, mismatched block, dense unstructured), selecting each on the
training split and scoring on a held-out split.
