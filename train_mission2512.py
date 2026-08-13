#!/usr/bin/env python3
"""Train and evaluate the TCV TorbeamNN using mission 2512 data only.

The split is performed by equilibrium frame within every shot so launcher-angle
samples from one equilibrium cannot leak between train, validation, and test.
This script intentionally has no dependency on TensorFlow or scikit-learn.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_NAMES = np.array(
    [
        "pol_ang", "tor_ang", "Bt_center", "Ip", "R0", "Z0", "aminor",
        "betan", "elong", "li", "volume", "ne_val_0", "ne_val_10",
        "ne_val_20",
    ]
)
INPUT_UNITS = np.array(
    [
        "rad", "rad", "T", "A", "m", "m", "cm", "1", "1", "1",
        "cm^3", "1e19 m^-3", "1e19 m^-3", "1e19 m^-3",
    ]
)
OUTPUT_NAMES = np.array(["rho_pol", "R", "Z", "CD_eta", "w_cd"])
OUTPUT_UNITS = np.array(["1", "m", "m", "1", "rho_pol"])


def discover_shots(data_dir: Path) -> tuple[int, ...]:
    """Return every complete shot pair and reject incomplete uploads."""
    input_pattern = re.compile(r"tbm_vector_shot_(\d+)\.mat$")
    output_pattern = re.compile(r"training_data_shot_(\d+)\.mat$")
    input_shots = {
        int(match.group(1))
        for path in data_dir.glob("tbm_vector_shot_*.mat")
        if (match := input_pattern.fullmatch(path.name))
    }
    output_shots = {
        int(match.group(1))
        for path in data_dir.glob("training_data_shot_*.mat")
        if (match := output_pattern.fullmatch(path.name))
    }
    missing_outputs = sorted(input_shots - output_shots)
    missing_inputs = sorted(output_shots - input_shots)
    if missing_outputs or missing_inputs:
        raise FileNotFoundError(
            "Incomplete mission data; missing output files for "
            f"{missing_outputs} and input files for {missing_inputs}"
        )
    if not input_shots:
        raise FileNotFoundError(f"No matching shot pairs found in {data_dir}")
    return tuple(sorted(input_shots))


def _finite_scalar(value: Any) -> float:
    result = float(value)
    return result if np.isfinite(result) else np.nan


def extract_shot(data_dir: Path, shot: int) -> dict[str, np.ndarray]:
    """Extract physical features, targets, validity, and frame indices."""
    input_path = data_dir / f"tbm_vector_shot_{shot}.mat"
    output_path = data_dir / f"training_data_shot_{shot}.mat"
    if not input_path.is_file() or not output_path.is_file():
        raise FileNotFoundError(f"Missing matching MAT files for shot {shot}")

    tbm = loadmat(input_path, struct_as_record=False, squeeze_me=True)[
        "tbm_vector_struct"
    ]
    outputs = loadmat(output_path, struct_as_record=False, squeeze_me=True)[
        "outputs"
    ]
    results = np.asarray(outputs.results, dtype=object)
    sequence = np.asarray(outputs.low_disc_seq, dtype=np.float64)
    if results.ndim != 2 or len(tbm) != results.shape[0]:
        raise ValueError(
            f"Shot {shot}: incompatible frames: tbm={len(tbm)}, "
            f"results={results.shape}"
        )
    if sequence.shape != (results.shape[1], 4):
        raise ValueError(
            f"Shot {shot}: low_disc_seq {sequence.shape} does not match "
            f"{results.shape[1]} points"
        )

    n_frames, n_points = results.shape
    features = np.full((n_frames * n_points, len(INPUT_NAMES)), np.nan)
    targets = np.full((n_frames * n_points, len(OUTPUT_NAMES)), np.nan)
    valid = np.zeros(n_frames * n_points, dtype=bool)
    frames = np.repeat(np.arange(n_frames), n_points)

    for frame in range(n_frames):
        eq = tbm[frame].inputs.eq_data
        prof = tbm[frame].inputs.prof_data
        density_profile = np.interp(
            [0.0, 0.5, 1.0],
            np.asarray(prof.rhopol, dtype=np.float64),
            np.asarray(prof.ne, dtype=np.float64),
        )
        base = np.array(
            [
                np.nan,
                np.nan,
                eq.B0,
                np.nan,
                eq.Rmaj / 100.0,
                eq.zA / 100.0,
                eq.Rmin,
                prof.betan,
                eq.kappa,
                eq.li,
                eq.volume * 1.0e6,
                np.nan,
                np.nan,
                np.nan,
            ],
            dtype=np.float64,
        )
        for point in range(n_points):
            row = frame * n_points + point
            density_factor = sequence[point, 2]
            features[row] = base
            features[row, 0] = np.deg2rad(sequence[point, 0])
            features[row, 1] = np.deg2rad(sequence[point, 1])
            features[row, 3] = eq.Ip * sequence[point, 3]
            features[row, 11:14] = density_profile * density_factor

            result = results[frame, point]
            if getattr(result, "exitflag", None) != 0:
                continue
            cd = getattr(result, "cd_profiles", None)
            if not hasattr(cd, "cd_deposition_width"):
                continue
            try:
                targets[row] = [
                    _finite_scalar(result.peak_absorption.rho_max),
                    _finite_scalar(result.peak_absorption.R),
                    _finite_scalar(result.peak_absorption.Z),
                    _finite_scalar(result.totals.ratio_cd),
                    _finite_scalar(cd.cd_deposition_width),
                ]
            except (AttributeError, TypeError, ValueError):
                continue
            valid[row] = (
                np.all(np.isfinite(features[row]))
                and np.all(np.isfinite(targets[row]))
                and 0.0 <= targets[row, 0] <= 1.0
                and -0.2 <= targets[row, 3] <= 0.2
                and targets[row, 4] > 0.0
            )

    return {
        "features": features[valid],
        "targets": targets[valid],
        "frames": frames[valid],
        "valid": valid,
        "n_total": np.array(len(valid)),
    }


def split_data(
    shot_data: dict[int, dict[str, np.ndarray]], seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, list[int]]]]:
    """Assign eight/one/one equilibrium frames to train/validation/test."""
    rng = np.random.default_rng(seed)
    assignments: dict[str, dict[str, list[int]]] = {}
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"features": [], "targets": [], "shots": [], "frames": []}
        for name in ("train", "validation", "test")
    }

    for shot, data in shot_data.items():
        frame_ids = np.unique(data["frames"])
        if len(frame_ids) < 3:
            raise ValueError(f"Shot {shot} has fewer than three valid frames")
        shuffled = rng.permutation(frame_ids)
        n_test = max(1, int(round(0.1 * len(shuffled))))
        n_validation = max(1, int(round(0.1 * len(shuffled))))
        assignment = {
            "train": shuffled[: -(n_validation + n_test)].tolist(),
            "validation": shuffled[-(n_validation + n_test) : -n_test].tolist(),
            "test": shuffled[-n_test:].tolist(),
        }
        assignments[str(shot)] = assignment
        for part, selected_frames in assignment.items():
            mask = np.isin(data["frames"], selected_frames)
            count = int(mask.sum())
            parts[part]["features"].append(data["features"][mask])
            parts[part]["targets"].append(data["targets"][mask])
            parts[part]["shots"].append(np.full(count, shot, dtype=np.int64))
            parts[part]["frames"].append(data["frames"][mask])

    combined: dict[str, dict[str, np.ndarray]] = {}
    for part, arrays in parts.items():
        combined[part] = {
            key: np.concatenate(value, axis=0) for key, value in arrays.items()
        }
    return combined, assignments


class TorbeamNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(14, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 5),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def as_tensor(values: np.ndarray) -> torch.Tensor:
    # PyTorch 1.13 in this environment predates NumPy 2's array API.
    return torch.tensor(values.tolist(), dtype=torch.float32)


def predict(model: nn.Module, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.asarray(model(as_tensor(values)).tolist(), dtype=np.float64)


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(OUTPUT_NAMES):
        residual = predictions[:, index] - targets[:, index]
        ss_res = float(np.sum(residual**2))
        centered = targets[:, index] - np.mean(targets[:, index])
        ss_tot = float(np.sum(centered**2))
        result[str(name)] = {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        }
    return result


def plot_results(
    output_dir: Path,
    history: dict[str, list[float]],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> None:
    epochs = np.arange(1, len(history["train_mse"]) + 1)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.semilogy(epochs, history["train_mse"], label="training")
    axis.semilogy(epochs, history["validation_mse"], label="validation")
    axis.axvline(np.argmin(history["validation_mse"]) + 1, color="k", ls="--", lw=1)
    axis.set(xlabel="Epoch", ylabel="Scaled MSE", title="Mission 2512 training history")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(fig)

    parity_metrics = metrics(targets, predictions)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
    for index, axis in enumerate(axes):
        truth = targets[:, index]
        estimate = predictions[:, index]
        axis.scatter(truth, estimate, s=8, alpha=0.42, edgecolors="none")
        low = min(float(truth.min()), float(estimate.min()))
        high = max(float(truth.max()), float(estimate.max()))
        padding = 0.03 * (high - low)
        axis.plot(
            [low - padding, high + padding], [low - padding, high + padding],
            "r--", lw=1.2,
        )
        axis.set(
            xlabel="TORBEAM",
            ylabel="Baseline 60×3 ReLU",
            title=str(OUTPUT_NAMES[index]),
            xlim=(low - padding, high + padding),
            ylim=(low - padding, high + padding),
        )
        values = parity_metrics[str(OUTPUT_NAMES[index])]
        axis.text(
            0.04,
            0.96,
            f"R² = {values['r2']:.4f}\nMAE = {values['mae']:.4g}\nRMSE = {values['rmse']:.4g}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round", "facecolor": "white", "alpha": 0.82,
                "edgecolor": "0.8",
            },
        )
        axis.grid(alpha=0.2)
    fig.suptitle("Baseline 60×3 ReLU held-out-frame parity — mission 2512")
    fig.tight_layout()
    fig.savefig(output_dir / "performance_parity.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for index, axis in enumerate(axes):
        residual = predictions[:, index] - targets[:, index]
        axis.scatter(targets[:, index], residual, s=8, alpha=0.45)
        axis.axhline(0.0, color="r", ls="--", lw=1)
        axis.set(xlabel="TORBEAM", ylabel="Prediction residual", title=str(OUTPUT_NAMES[index]))
        axis.grid(alpha=0.2)
    fig.suptitle("Mission 2512 held-out-frame residuals")
    fig.tight_layout()
    fig.savefig(output_dir / "performance_residuals.png", dpi=180)
    plt.close(fig)


def save_model(
    output_path: Path,
    model: TorbeamNet,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "input_names": INPUT_NAMES,
        "input_units": INPUT_UNITS,
        "output_names": OUTPUT_NAMES,
        "output_units": OUTPUT_UNITS,
    }
    linear_layers = [module for module in model.layers if isinstance(module, nn.Linear)]
    for index, layer in enumerate(linear_layers):
        # Store kernels in Keras/NumPy convention: (input, output).
        arrays[f"kernel_{index}"] = np.asarray(
            layer.weight.detach().tolist(), dtype=np.float32
        ).T
        arrays[f"bias_{index}"] = np.asarray(
            layer.bias.detach().tolist(), dtype=np.float32
        )
    np.savez_compressed(output_path, **arrays)


def write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    metrics_text = []
    for name, unit in zip(OUTPUT_NAMES, OUTPUT_UNITS):
        values = report["metrics"]["test"][str(name)]
        label = str(name) if unit == "1" else f"{name} ({unit})"
        metrics_text.append(
            f"| {label} | {values['r2']:.4f} | {values['mae']:.6g} | "
            f"{values['rmse']:.6g} |"
        )
    shots_text = ", ".join(map(str, report["shots"]))
    counts = report["sample_counts"]
    readme = f"""# Mission 2512 TCV TorbeamNN

This model was trained exclusively on all complete shot pairs detected in
`torbeam_training_data_mission2512`: {shots_text}. No samples from the earlier
mission were used.

The model is a `14-60-60-60-5` dense ReLU network. Each shot's equilibrium
frames were split into 80% training, 10% validation and 10% untouched testing.
Input and output scalers were fitted on the training frames only. Training
stopped at epoch {report['stopped_epoch']} and restored the best checkpoint
from epoch {report['best_epoch']}.

Valid sample counts were {counts['train']:,} training, {counts['validation']:,}
validation and {counts['test']:,} test. Aggregate held-out-frame results are:

| Output | R2 | MAE | RMSE |
|---|---:|---:|---:|
{chr(10).join(metrics_text)}

`model.npz` stores the network weights and training-only scalers in a
dependency-free format. `training_report.json` contains the full train,
validation and test metrics, validity counts, frame assignment and per-shot
test metrics. `predictions.npz` and `history.npz` preserve the evaluation data
and optimization history. The PNG files visualize convergence, parity and
residuals.

Retrain from the repository root with:

```bash
MPLCONFIGDIR=/tmp/mplconfig python train_mission2512.py
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = discover_shots(data_dir)
    print(f"Data directory (exclusive): {data_dir}", flush=True)
    print(f"Shots: {', '.join(map(str, shots))}", flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))

    shot_data: dict[int, dict[str, np.ndarray]] = {}
    validity: dict[str, dict[str, int | float]] = {}
    for shot in shots:
        data = extract_shot(data_dir, shot)
        shot_data[shot] = data
        n_total = int(data["n_total"])
        n_valid = len(data["targets"])
        validity[str(shot)] = {
            "total": n_total,
            "valid": n_valid,
            "valid_percent": 100.0 * n_valid / n_total,
        }
        print(
            f"Shot {shot}: {n_valid}/{n_total} valid samples "
            f"({100.0 * n_valid / n_total:.2f}%)",
            flush=True,
        )

    parts, assignments = split_data(shot_data, args.seed)
    x_mean = parts["train"]["features"].mean(axis=0)
    x_scale = parts["train"]["features"].std(axis=0)
    y_mean = parts["train"]["targets"].mean(axis=0)
    y_scale = parts["train"]["targets"].std(axis=0)
    if np.any(x_scale == 0.0) or np.any(y_scale == 0.0):
        raise ValueError("A training feature or target has zero variance")

    scaled: dict[str, dict[str, np.ndarray]] = {}
    for name, data in parts.items():
        scaled[name] = {
            "features": (data["features"] - x_mean) / x_scale,
            "targets": (data["targets"] - y_mean) / y_scale,
        }
        print(f"{name.capitalize()} samples: {len(data['targets'])}", flush=True)

    train_x = as_tensor(scaled["train"]["features"])
    train_y = as_tensor(scaled["train"]["targets"])
    validation_x = as_tensor(scaled["validation"]["features"])
    validation_y = as_tensor(scaled["validation"]["targets"])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = TorbeamNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    history: dict[str, list[float]] = {"train_mse": [], "validation_mse": []}
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_squared_error = 0.0
        element_count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)
            loss.backward()
            optimizer.step()
            sum_squared_error += float(loss.item()) * batch_y.numel()
            element_count += batch_y.numel()
        train_loss = sum_squared_error / element_count
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y).item())
        history["train_mse"].append(train_loss)
        history["validation_mse"].append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:4d}: train MSE={train_loss:.6f}, "
                f"validation MSE={validation_loss:.6f}, best={best_loss:.6f}",
                flush=True,
            )
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)

    all_predictions: dict[str, np.ndarray] = {}
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    per_shot_test_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name, data in parts.items():
        prediction_scaled = predict(model, scaled[name]["features"])
        prediction = prediction_scaled * y_scale + y_mean
        all_predictions[name] = prediction
        all_metrics[name] = metrics(data["targets"], prediction)
    for shot in shots:
        mask = parts["test"]["shots"] == shot
        per_shot_test_metrics[str(shot)] = metrics(
            parts["test"]["targets"][mask], all_predictions["test"][mask]
        )

    report = {
        "data_directory": str(data_dir),
        "shots": list(shots),
        "architecture": [14, 60, 60, 60, 5],
        "activation": "relu",
        "output_activation": "linear",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eta_cd_filter": [-0.2, 0.2],
        "split_unit": "equilibrium frame",
        "best_epoch": best_epoch,
        "stopped_epoch": len(history["train_mse"]),
        "best_validation_scaled_mse": best_loss,
        "validity": validity,
        "sample_counts": {name: len(data["targets"]) for name, data in parts.items()},
        "frame_assignment": assignments,
        "metrics": all_metrics,
        "test_metrics_by_shot": per_shot_test_metrics,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_readme(output_dir, report)

    save_model(output_dir / "model.npz", model, x_mean, x_scale, y_mean, y_scale)
    np.savez_compressed(
        output_dir / "history.npz",
        epoch=np.arange(1, len(history["train_mse"]) + 1),
        train_mse=np.asarray(history["train_mse"]),
        validation_mse=np.asarray(history["validation_mse"]),
    )
    prediction_arrays: dict[str, np.ndarray] = {}
    for name, data in parts.items():
        prediction_arrays[f"{name}_features"] = data["features"]
        prediction_arrays[f"{name}_targets"] = data["targets"]
        prediction_arrays[f"{name}_predictions"] = all_predictions[name]
        prediction_arrays[f"{name}_shots"] = data["shots"]
        prediction_arrays[f"{name}_frames"] = data["frames"]
    np.savez_compressed(output_dir / "predictions.npz", **prediction_arrays)
    plot_results(
        output_dir,
        history,
        parts["test"]["targets"],
        all_predictions["test"],
    )

    print(f"Best epoch: {best_epoch}", flush=True)
    print("Held-out test metrics:", flush=True)
    for output_name in OUTPUT_NAMES:
        values = all_metrics["test"][str(output_name)]
        print(
            f"  {output_name:7s} R2={values['r2']: .4f}  "
            f"MAE={values['mae']:.6g}  RMSE={values['rmse']:.6g}",
            flush=True,
        )
    print(f"Artifacts written to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("torbeam_training_data_mission2512"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tcv_torbeamnn_mission2512")
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=6.3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
