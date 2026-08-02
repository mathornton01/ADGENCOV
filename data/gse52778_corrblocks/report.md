# ADGENCOV: GSE52778 run summary

- matrix: 16 samples x 64 genes
- symmetry: `correlation_blocks`, group sizes [2, 15, 20, 27]
- criterion: exact leave-one-out Gaussian NLL
- candidates scored: 40
- recommended: `ad_target_ridge`

| Rank | Method | Params | LOO-NLL |
|---:|---|---|---:|
| 1 | `ad_target_ridge` | lam=0.7 | 84.998 |
| 2 | `ridge` | alpha=0.7 | 85.165 |
| 3 | `oas` | -- | 85.365 |
| 4 | `ad_target_oas` | -- | 85.588 |
| 5 | `lw` | -- | 86.082 |
| 6 | `ad_target_lw` | -- | 86.722 |
| 7 | `ad_target_ridge` | lam=0.9 | 86.793 |
| 8 | `ad_target_optimal` | -- | 86.954 |
| 9 | `ad_target_ridge` | lam=0.5 | 87.368 |
| 10 | `ridge` | alpha=0.4 | 88.663 |
| 11 | `ad_ridge` | alpha=0.4 | 91.132 |
| 12 | `ad_linear_lw` | -- | 91.192 |
