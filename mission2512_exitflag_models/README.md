# Mission 2512 exit-flag prediction

> The subsequent five-category retraining that combines flags 3+8 as
> **cutoff** and 4+7+10+100 as **failed run** is in
> `grouped_exit_flags/`, with its own models, results, figures, and README.

This folder contains two multiclass classifiers that predict the TORBEAM exit
flag from inputs available **before** running TORBEAM. Result fields,
trajectories, absorption profiles, output-file presence, runtime, and all
exit-derived values are excluded.

## Exit-code semantics

The supplied TORBEAM guide defines the base flags in `rhoresult(19)`:

| Flag | Meaning |
|---:|---|
| 0 | Normal exit with non-zero absorption |
| 1 | No plasma intersection |
| 2 | Plasma crossed without absorption |
| 3 | Plasma intersected; cutoff |
| 4 | Integrator failure |
| 5 | Negative input density or temperature |
| 6 | Magnetic axis not found |
| 7 | Maximum integration-step count exceeded |
| 8 | Cutoff at the vacuum/plasma boundary |

The MATLAB layer that produced these MAT files adds two dataset-specific
meanings: `torbeam_output_ech.m` changes flag 0 to **10** when `rho_max == -1`
(invalid absorption point), and assigns **100** when no `rhoresult` output file
is returned (timeout/no output). This wrapper meaning of 100 is the relevant
one here; the PDF separately reserves +100/+200 as raw allocation/deallocation
modifiers.

Mission 2512 contains 33,000 trials. Counts are 26,254 / 134 / 544 / 5,555 /
1 / 51 / 361 / 100 for flags 0 / 1 / 2 / 3 / 4 / 8 / 10 / 100. No examples
of flags 5, 6, or 7 occur.

## Models

**Small model — compact decision tree**

- 63 pre-run inputs, including Fourier angle encodings, trial multipliers,
  equilibrium scalars, and five-point density/temperature/rhotor profiles.
- Depth 8, 347 nodes in the evaluation fit, tempered class weights.
- Deployable file: `models/small_decision_tree.joblib` (about 20 kB).

**Best overall model — profile MLP**

- 111 pre-run inputs, using 21-point profiles but not the coarse equilibrium
  maps, which did not improve the neural validation score.
- Hidden widths 256–256–128 with SiLU, batch normalization, dropout, and
  tempered inverse-frequency cross-entropy.
- A single validation-selected factor of 0.55 is applied to every nonzero
  class probability before the final argmax. This offsets the deliberately
  failure-heavy training loss.
- Deployable file: `models/best_profile_mlp.pt` (about 555 kB).

Both deployable files are refit on train+validation. The `_evaluation` files
retain the exact training-only fits used to calculate the held-out results.

## Held-out result

The split holds out one complete equilibrium frame per shot for validation and
one for test. Configurations and the probability factor were locked before the
3,300 test rows were read.

| Model | Exact accuracy | Observed-class macro F1 | Weighted F1 | Failure precision | Failure recall | Failure PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Always flag 0 | 0.8197 | 0.1502 | 0.7385 | 0 | 0 | 0.1803 |
| Small tree | 0.9103 | **0.5186** | 0.9165 | 0.7716 | 0.9025 | 0.9151 |
| Best MLP | **0.9418** | 0.4560 | **0.9412** | **0.9003** | 0.8958 | **0.9707** |

The MLP is the stronger operational model: it improves exact accuracy by 3.15
percentage points over the small tree and has much better failure ranking and
precision. The small tree wins test macro F1 because it identifies 5 of the
only 10 flag-8 examples; the MLP identifies none. Flags 1 and 4 have zero test
support. Consequently, macro F1 is very sensitive to a handful of rare cases
and does not establish that the small tree is generally better.

Cluster-bootstrap 95% intervals over the 11 held-out shot/frame groups are:

| Model | Exact accuracy | Macro F1 | Failure F1 | Failure PR-AUC |
|---|---|---|---|---|
| Small tree | 0.894–0.928 | 0.447–0.563 | 0.741–0.884 | 0.825–0.956 |
| Best MLP | 0.926–0.958 | 0.432–0.573 | 0.807–0.943 | 0.916–0.989 |

Exact prediction of every mechanism is not statistically supportable from this
dataset alone: flag 4 has one training example; flags 1, 8, and 100 are rare
and localized to few shots/frames. For a control decision, use the MLP's
`failure_probability = 1 - P(flag=0)` and treat the exact rare-code prediction
as diagnostic rather than authoritative.

There are only two 300-point low-discrepancy designs (600 unique trial vectors)
reused across frames. A frame-held-out test therefore measures new equilibria
at mostly familiar trial coordinates. `dataset_report.json` records the exact
frame assignments. For arbitrary new coordinates or new shots, use crossed
frame-and-design-vector validation before deployment; shot 85933 is also the
only positive-equilibrium-sign shot and is unsupported extrapolation in a
leave-one-shot-out test.

## Reproduce and use

```bash
python build_dataset.py
python search_models.py --stage small
python search_models.py --stage forest
python search_models.py --stage lgb
python search_models.py --stage xgb
python search_neural.py
python finalize_models.py
```

Generate predictions from a cached split:

```bash
python predict_models.py --model best --input dataset_cache.npz \
  --split validation --output validation_predictions.csv
```

For an external NPZ, provide a 63-column compact matrix to the small model or a
111-column profile matrix to the MLP and name it with `--key`. Feature order is
stored in each model artifact and in `dataset_report.json`.

The saved search contains 122 successful candidates: decision trees, Extra
Trees, histogram gradient boosting, LightGBM, XGBoost, and four MLPs. Separate
linear-logistic probes were inferior and were not retained. Eight
histogram-boosting attempts failed before its internal early split
was disabled because the singleton flag-4 class cannot be stratified.
`results/validation_search_summary.csv` contains every completed/error record.

Key outputs:

- `results/final_metrics.json`: complete validation/test metrics, confusion
  matrices, per-shot results, and clustered confidence intervals.
- `results/test_summary.csv` and `results/test_per_class.csv`: compact tables.
- `results/test_predictions.npz`: row-level held-out predictions.
- `figures/`: PNG and vector-PDF comparisons, confusion matrices, per-class F1,
  failure precision–recall, and class balance.
