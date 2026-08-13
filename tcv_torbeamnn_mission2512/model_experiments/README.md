# Best mission-2512 surrogate

The selected model is a shared `128-128-64` SiLU trunk with a separate `32-1`
head for each of the five physical outputs. It was trained on all complete shot
pairs in `../../torbeam_training_data_mission2512` with a frame-level split.

- `multihead_silu.npz`: network state and training-only scalers.
- `best_model_performance_parity.png`: held-out predictions with R², MAE and
  RMSE for every output.
- `best_model_vs_baseline.png`: held-out R² comparison with the previous model.
- `../baseline_complete_data_r2.png`: full five-output R² parity figure for the
  baseline retrained on all 11 mission-2512 shots.

Retrain the selected architecture from raw data with:

```bash
MPLCONFIGDIR=/tmp/mplconfig python best_model_mission2512.py
```
