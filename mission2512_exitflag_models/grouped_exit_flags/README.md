# Mission 2512 grouped exit-label models

This folder contains the requested retraining of only the two retained model
architectures. The raw mission-2512 labels are grouped as follows:

| Group | Original flags | Meaning |
|---|---|---|
| Normal | 0 | Normal exit |
| No intersection | 1 | No plasma intersection |
| Crossed, no absorption | 2 | Plasma crossed without absorption |
| Cutoff | 3, 8 | Plasma or vacuum-boundary cutoff |
| Failed run | 4, 7, 10, 100 | Integrator/max-step/wrapper-invalid-point/timeout failure |

Flags 5 and 6 are not assigned because they were not included in the requested
grouping and do not occur in mission 2512. Flag 7 also does not occur, and flag
4 occurs once. Consequently, the learned failed-run group is almost entirely
based on flags 10 and 100.

## Retained architectures

- **Small tree:** the same 63-input compact feature set and depth-8 decision
  tree with a minimum of five training rows per leaf. The grouped validation
  sweep selected class-weight power 0.20 and no probability correction.
- **Best profile MLP:** the same 111-input profile feature set and
  256–256–128 hidden widths with batch normalization, SiLU, and dropout 0.12.
  The grouped validation sweep selected class-weight power 0.25, epoch 73, and
  a nearly neutral 1.025 multiplier on all non-normal probabilities.

Both models were retrained from scratch with five output units. The old
eight-output probabilities were not merely added together. As before, every
input is available before running TORBEAM; result fields and output-presence
fields are excluded.

## Held-out-frame results

Configurations were selected using the 3,300 validation rows, then evaluated
on the same 3,300 frame-held-out test rows retained by the earlier experiment.
The test was excluded from fitting and grouped-model selection, although it had
already been inspected in the preceding eight-code study.

| Test metric | Majority baseline | Small tree | Best MLP |
|---|---:|---:|---:|
| Grouped accuracy | 0.8197 | 0.9358 | **0.9461** |
| Macro F1, four supported test groups | 0.2252 | 0.6089 | **0.6510** |
| Macro F1, all five groups | 0.1802 | 0.4872 | **0.5208** |
| Weighted F1 | 0.7385 | 0.9349 | **0.9448** |
| Non-normal PR-AUC | 0.1803 | 0.9343 | **0.9588** |
| Failed-run PR-AUC | 0.0179 | 0.1199 | **0.2533** |

Per-group test performance:

| Group (test support) | Small tree F1 | Best MLP F1 |
|---|---:|---:|
| Normal (2,705) | 0.9712 | **0.9750** |
| No intersection (0) | N/A | N/A |
| Crossed, no absorption (50) | 0.4071 | **0.4444** |
| Cutoff (486) | 0.8793 | **0.9100** |
| Failed run (59) | 0.1782 | **0.2745** |

The MLP identifies 435/486 cutoff cases and 14/59 failed runs. Failed-run
ranking is useful relative to its 1.79% prevalence, but exact failed-run recall
at the five-way argmax is only 23.7%; use the continuous failed-run probability
when screening risky runs.

Shot/frame cluster-bootstrap 95% intervals:

| Metric | Small tree | Best MLP |
|---|---|---|
| Accuracy | 0.922–0.950 | 0.928–0.962 |
| Supported-group macro F1 | 0.550–0.656 | 0.599–0.702 |
| Failed-run PR-AUC | 0.079–0.195 | 0.186–0.426 |

The test split has no no-intersection samples despite 134 such examples in the
full data, so its test F1 is shown as N/A. The reported 0.651 MLP macro F1 is
over the four groups actually represented; the corresponding all-five score is
0.521. Only 600 distinct low-discrepancy trial vectors are reused across
frames, so this remains an interpolation-oriented frame test rather than proof
of new-shot or arbitrary-coordinate generalization.

## Files

Deployable models, refit on train+validation with test excluded:

- `models/small_grouped_tree.joblib` (about 22 kB)
- `models/best_grouped_profile_mlp.pt` (about 554 kB)

The `_evaluation` artifacts retain the exact training-only fits used for the
reported test numbers. `results/final_metrics.json` contains full validation,
test, per-shot, confusion-matrix, and bootstrap results. Row-level test
probabilities are in `results/test_predictions.npz`.

Adapted figures are available as PNG and vector PDF:

- `grouped_model_comparison`
- `grouped_confusion_matrices`
- `grouped_per_class_f1`
- `grouped_failed_run_precision_recall`
- `grouped_class_distribution`

To reproduce:

```bash
python train_validate_grouped.py
python evaluate_grouped.py
```

To run a deployable model on a feature matrix:

```bash
python predict_grouped.py --model best \
  --input ../dataset_cache.npz --split validation \
  --output validation_grouped_predictions.csv
```

An external NPZ must provide the same ordered 63-column compact matrix for the
tree or 111-column profile matrix for the MLP. Exact feature names are stored in
the model artifacts.

