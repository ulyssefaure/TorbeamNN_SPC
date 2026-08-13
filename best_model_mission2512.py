#!/usr/bin/env python3
"""Train the selected multi-head SiLU surrogate on mission-2512 data."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_mission2512 import (
    INPUT_NAMES,
    INPUT_UNITS,
    OUTPUT_NAMES,
    OUTPUT_UNITS,
    as_tensor,
    discover_shots,
    extract_shot,
    metrics,
    split_data,
)


class MultiHeadSiLU(nn.Module):
    """Shared 128-128-64 trunk with one 32-unit head per physical output."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(len(INPUT_NAMES), 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1)) for _ in OUTPUT_NAMES]
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(features)
        return torch.cat([head(shared) for head in self.heads], dim=1)


def predict(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.asarray(model(as_tensor(features)).tolist(), dtype=np.float64)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
) -> None:
    config = {
        "architecture": "shared 128-128-64 SiLU trunk; five independent 32-1 SiLU heads",
        "optimizer": "AdamW",
        "learning_rate": 5.0e-4,
        "weight_decay": 1.0e-5,
        "batch_size": 128,
        "split_seed": 42,
        "model_seed": 104,
    }
    arrays: dict[str, np.ndarray] = {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "input_names": INPUT_NAMES,
        "input_units": INPUT_UNITS,
        "output_names": OUTPUT_NAMES,
        "output_units": OUTPUT_UNITS,
        "config_json": np.array(json.dumps(config, sort_keys=True)),
    }
    for key, value in model.state_dict().items():
        arrays[f"state__{key}"] = np.asarray(value.detach().tolist(), dtype=np.float32)
    np.savez_compressed(path, **arrays)


def plot_r2(path: Path, targets: np.ndarray, predictions: np.ndarray) -> None:
    scores = metrics(targets, predictions)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
    for index, (name, axis) in enumerate(zip(OUTPUT_NAMES, axes)):
        truth = targets[:, index]
        estimate = predictions[:, index]
        low = min(float(truth.min()), float(estimate.min()))
        high = max(float(truth.max()), float(estimate.max()))
        padding = 0.03 * (high - low)
        axis.scatter(truth, estimate, s=8, alpha=0.42, edgecolors="none")
        axis.plot([low - padding, high + padding], [low - padding, high + padding], "r--", lw=1.2)
        axis.set(
            xlabel="TORBEAM",
            ylabel="Multi-head SiLU",
            title=str(name),
            xlim=(low - padding, high + padding),
            ylim=(low - padding, high + padding),
        )
        values = scores[str(name)]
        axis.text(
            0.04, 0.96,
            f"R² = {values['r2']:.4f}\nMAE = {values['mae']:.4g}\nRMSE = {values['rmse']:.4g}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
        )
        axis.grid(alpha=0.2)
    fig.suptitle("Multi-head SiLU held-out-frame parity — mission 2512", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = discover_shots(data_dir)
    torch.set_num_threads(max(1, args.threads))
    np.random.seed(42)
    torch.manual_seed(104)

    shot_data = {shot: extract_shot(data_dir, shot) for shot in shots}
    parts, _ = split_data(shot_data, seed=42)
    x_mean = parts["train"]["features"].mean(axis=0)
    x_scale = parts["train"]["features"].std(axis=0)
    y_mean = parts["train"]["targets"].mean(axis=0)
    y_scale = parts["train"]["targets"].std(axis=0)
    scaled = {
        split: {
            "features": (data["features"] - x_mean) / x_scale,
            "targets": (data["targets"] - y_mean) / y_scale,
        }
        for split, data in parts.items()
    }

    model = MultiHeadSiLU()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1.0e-5
    )
    loss_function = nn.MSELoss()
    validation_x = as_tensor(scaled["validation"]["features"])
    validation_y = as_tensor(scaled["validation"]["targets"])
    loader = DataLoader(
        TensorDataset(as_tensor(scaled["train"]["features"]), as_tensor(scaled["train"]["targets"])),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(104),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, 301):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y).item())
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch:3d}: validation scaled MSE={validation_loss:.6f}", flush=True)
        if stale >= 70:
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)

    test_scaled = predict(model, scaled["test"]["features"])
    test_predictions = test_scaled * y_scale + y_mean
    test_metrics = metrics(parts["test"]["targets"], test_predictions)
    save_checkpoint(output_dir / "multihead_silu.npz", model, x_mean, x_scale, y_mean, y_scale)
    plot_r2(output_dir / "best_model_performance_parity.png", parts["test"]["targets"], test_predictions)
    print(f"Best epoch: {best_epoch}; validation scaled MSE: {best_loss:.6f}")
    for name in OUTPUT_NAMES:
        values = test_metrics[str(name)]
        print(f"{name}: R2={values['r2']:.4f}, MAE={values['mae']:.6g}, RMSE={values['rmse']:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("torbeam_training_data_mission2512"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tcv_torbeamnn_mission2512/model_experiments")
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
