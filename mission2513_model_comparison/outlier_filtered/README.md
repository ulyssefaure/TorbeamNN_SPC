# Mission 2513 input-domain-filtered evaluation

The original data and results remain unchanged. This evaluation excludes a
test sample only when at least one of its 51 wide Fourier-profile inputs is more
than 8 training standard deviations from the training mean. The
rule uses training inputs only--not test targets, prediction errors, or R2.

Exactly one of 268 test samples is flagged: shot 85811, frame 7, launcher point
17. Its maximum input-domain score is 17.22 sigma; ten
density-profile coordinates are 10.15--17.22 sigma from training and exceed
the corresponding training maxima. No training or validation sample exceeds
the 8-sigma threshold. The result is not sensitive to the precise cutoff: any
threshold from 6.25 through 17.21 isolates the same point. Since the excluded point is in the test set, retraining
would not change either checkpoint; only the in-domain evaluation changes.

| Output | Article R2 | Wide Fourier-profile R2 | Difference |
|---|---:|---:|---:|
| `rho_pol` | 0.888390 | 0.926016 | +0.037626 |
| `R` | 0.855488 | 0.887494 | +0.032007 |
| `Z` | 0.843361 | 0.896268 | +0.052906 |
| `CD_eta` | 0.840624 | 0.933120 | +0.092496 |
| `w_cd` | 0.210564 | 0.280786 | +0.070223 |

See `outlier_report.json` for the exact rule, excluded values, original and
filtered metrics, and squared-error influence. The raw comparison in the parent
folder should still be reported whenever out-of-domain behavior matters.
