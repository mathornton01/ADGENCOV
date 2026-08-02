# ADGENCOV: GSE52778 run summary

- matrix: 16 samples x 64 genes
- symmetry: `gene_family`, group sizes [1, 1, 1, 1, 29, 31]
- criterion: exact leave-one-out Gaussian NLL
- candidates scored: 40
- recommended: `ad_target_ridge`

| Rank | Method | Params | LOO-NLL |
|---:|---|---|---:|
| 1 | `ad_target_ridge` | lam=0.7 | 80.452 |
| 2 | `ad_target_ridge` | lam=0.9 | 82.250 |
| 3 | `ad_target_ridge` | lam=0.5 | 82.553 |
| 4 | `ridge` | alpha=0.7 | 85.165 |
| 5 | `oas` | -- | 85.365 |
| 6 | `lw` | -- | 86.082 |
| 7 | `ad_target_oas` | -- | 86.965 |
| 8 | `ad_elastic_net` | l1_ratio=0.25, lam=0.01 | 87.325 |
| 9 | `ad_lasso` | lam=0.01 | 87.606 |
| 10 | `ad_elastic_net` | l1_ratio=0.25, lam=0.03 | 88.162 |
| 11 | `ad_ridge` | alpha=0.05 | 88.530 |
| 12 | `ridge` | alpha=0.4 | 88.663 |
