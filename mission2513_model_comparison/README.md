# Mission 2513: article model versus wide Fourier-profile network

Both models were trained from scratch exclusively on the 11 complete shot pairs
in `../torbeam_training_data_mission2513`. They use the same equilibrium-frame
split (seed 42), with 3,611 training, 795
validation, and 268 untouched test samples. Scalers were fitted
on training samples only, and checkpoints were selected only by validation MSE.

Of 33,000 raw TORBEAM evaluations, 4,674 pass the established convergence and
physical-validity filters. Successful evaluations are unevenly distributed
between frames, so the whole-frame split produces a relatively small 268-sample
test partition. The comparison is controlled because both models use exactly
the same saved frames and samples, but the absolute R2 values should be regarded
as specific to this fixed split.

| Output | Article 60x3 ReLU R2 | Wide Fourier-profile R2 | Difference |
|---|---:|---:|---:|
| `rho_pol` | 0.580140 | 0.802291 | +0.222151 |
| `R` | 0.356632 | 0.801434 | +0.444802 |
| `Z` | 0.770169 | 0.893250 | +0.123081 |
| `CD_eta` | 0.839986 | 0.657333 | -0.182653 |
| `w_cd` | 0.051720 | 0.201551 | +0.149830 |

Aggregate scaled MSE:

| Model | Validation | Held-out test |
|---|---:|---:|
| Article 60x3 ReLU | 0.385667 | 0.860309 |
| Wide Fourier-profile NN | 0.231181 | 0.470701 |

Artifacts:

- `article_model.npz` and `wide_fourier_profile_model.npz`: weights and training-only scalers.
- `predictions.npz`: targets and predictions for all partitions.
- `training_report.json`: data assignments, configurations, histories, and metrics.
- `article_r2.png` and `wide_fourier_r2.png`: standalone parity/R2 charts.
- `article_vs_wide_fourier_r2.png`: vertically aligned parity/R2 comparison.
- `r2_bar_comparison.png`: direct output-wise R2 comparison.
- `training_histories.png`: convergence and selected epochs.

An additional training-input-domain audit is available in `outlier_filtered/`.
It preserves these raw results and separately reports an in-domain evaluation
after a conservative input-only screen.

Reproduce from the repository root with:

```bash
MPLCONFIGDIR=/tmp/mission2513_mpl python mission2513_model_comparison/train_models.py
```
