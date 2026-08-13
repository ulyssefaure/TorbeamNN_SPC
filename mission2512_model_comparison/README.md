# Mission 2512 article and wide-network comparison

Both models were retrained using the same frame split after applying the
training-input-only 8-sigma screen documented in `training_report.json`. The
screen removes 6 of 20,801 training samples, no validation samples, and 1 of
2,672 test samples. Chart titles intentionally contain no filtering annotation.

| Output | Article R2 | Article MAE | Wide R2 | Wide MAE |
|---|---:|---:|---:|---:|
| `rho_pol` | 0.962145 | 0.0244865 | 0.971869 | 0.0190298 |
| `R` | 0.888018 | 0.00215528 | 0.916275 | 0.00183072 |
| `Z` | 0.978276 | 0.00857011 | 0.983961 | 0.00626932 |
| `CD_eta` | 0.899648 | 0.00375602 | 0.976995 | 0.00203262 |
| `w_cd` | 0.627392 | 0.0283398 | 0.675624 | 0.0257731 |

MAE is reported in each output's native units: `R` and `Z` in metres,
`rho_pol` and `CD_eta` dimensionless, and `w_cd` in normalized poloidal-radius units.

The CSV and LaTeX forms of this table are `metrics_table.csv` and
`metrics_table.tex`. Standalone and stacked parity figures, an R2 bar chart,
models, predictions, and the complete report are stored in this folder.

Reproduce with:

```bash
MPLCONFIGDIR=/tmp/mission2512_screened_mpl python mission2512_model_comparison/train_screened_models.py
```
