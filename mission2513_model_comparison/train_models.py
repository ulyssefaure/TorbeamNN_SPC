#!/usr/bin/env python3
"""Train the article and wide Fourier-profile networks on mission 2513 only."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = np.array(["rho_pol", "R", "Z", "CD_eta", "w_cd"])
RHO11 = np.linspace(0.0, 1.0, 11)
ARTICLE_NAMES = np.array([
    "pol_ang", "tor_ang", "Bt_center", "Ip", "R0", "Z0", "aminor",
    "betan", "elong", "li", "volume", "ne_rho_0", "ne_rho_0.5", "ne_rho_1",
])
PROFILE_NAMES = np.array([
    "pol_ang", "tor_ang", "Bt_center", "Ip_scaled", "R0", "Z0", "aminor",
    "betan", "elong", "li", "volume", "density_factor", "current_factor",
] + [f"ne_rho_{rho:.1f}" for rho in RHO11]
  + [f"Te_rho_{rho:.1f}" for rho in RHO11])
FOURIER_NAMES = np.array([
    f"{function}_{angle}_k{harmonic}"
    for harmonic in range(1, 5)
    for angle in ("pol", "tor")
    for function in ("sin", "cos")
])
WIDE_NAMES = np.concatenate([PROFILE_NAMES, FOURIER_NAMES])


def as_tensor(values: np.ndarray) -> torch.Tensor:
    # The local PyTorch build predates NumPy 2's array API.
    return torch.tensor(values.tolist(), dtype=torch.float32)


def discover_shots(data_dir: Path) -> tuple[int, ...]:
    input_pattern = re.compile(r"tbm_vector_shot_(\d+)\.mat$")
    output_pattern = re.compile(r"training_data_shot_(\d+)\.mat$")
    inputs = {
        int(match.group(1)) for path in data_dir.glob("tbm_vector_shot_*.mat")
        if (match := input_pattern.fullmatch(path.name))
    }
    outputs = {
        int(match.group(1)) for path in data_dir.glob("training_data_shot_*.mat")
        if (match := output_pattern.fullmatch(path.name))
    }
    if inputs != outputs:
        raise ValueError(
            f"Incomplete pairs: inputs-only={sorted(inputs-outputs)}, "
            f"outputs-only={sorted(outputs-inputs)}"
        )
    if not inputs:
        raise ValueError(f"No complete shot pairs found in {data_dir}")
    return tuple(sorted(inputs))


def finite(value: object) -> float:
    result = float(value)
    return result if np.isfinite(result) else np.nan


def extract_shot(data_dir: Path, shot: int) -> dict[str, np.ndarray]:
    tbm = loadmat(
        data_dir / f"tbm_vector_shot_{shot}.mat",
        struct_as_record=False,
        squeeze_me=True,
    )["tbm_vector_struct"]
    outputs = loadmat(
        data_dir / f"training_data_shot_{shot}.mat",
        struct_as_record=False,
        squeeze_me=True,
    )["outputs"]
    results = np.asarray(outputs.results, dtype=object)
    sequence = np.asarray(outputs.low_disc_seq, dtype=np.float64)
    if results.ndim != 2 or len(tbm) != results.shape[0]:
        raise ValueError(f"Shot {shot}: incompatible frame shapes")
    if sequence.shape != (results.shape[1], 4):
        raise ValueError(f"Shot {shot}: incompatible low-discrepancy sequence")

    article_rows: list[np.ndarray] = []
    wide_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frames: list[int] = []
    points: list[int] = []
    n_frames, n_points = results.shape

    for frame in range(n_frames):
        inputs = tbm[frame].inputs
        eq, prof = inputs.eq_data, inputs.prof_data
        rho = np.asarray(prof.rhopol, dtype=np.float64)
        ne11 = np.interp(RHO11, rho, np.asarray(prof.ne, dtype=np.float64))
        te11 = np.interp(RHO11, rho, np.asarray(prof.te, dtype=np.float64))
        for point in range(n_points):
            result = results[frame, point]
            if getattr(result, "exitflag", None) != 0:
                continue
            cd = getattr(result, "cd_profiles", None)
            if not hasattr(cd, "cd_deposition_width"):
                continue
            try:
                target = np.array([
                    finite(result.peak_absorption.rho_max),
                    finite(result.peak_absorption.R),
                    finite(result.peak_absorption.Z),
                    finite(result.totals.ratio_cd),
                    finite(cd.cd_deposition_width),
                ])
            except (AttributeError, TypeError, ValueError):
                continue
            if not (
                np.all(np.isfinite(target))
                and 0.0 <= target[0] <= 1.0
                and -0.2 <= target[3] <= 0.2
                and target[4] > 0.0
            ):
                continue

            density_factor = float(sequence[point, 2])
            current_factor = float(sequence[point, 3])
            if not np.isfinite(density_factor) or density_factor <= 0.0:
                continue
            pol_angle = float(np.deg2rad(sequence[point, 0]))
            tor_angle = float(np.deg2rad(sequence[point, 1]))
            core = np.array([
                pol_angle, tor_angle, eq.B0, eq.Ip * current_factor,
                eq.Rmaj / 100.0, eq.zA / 100.0, eq.Rmin, prof.betan,
                eq.kappa, eq.li, eq.volume * 1.0e6,
            ], dtype=np.float64)
            article = np.concatenate([
                core,
                np.interp([0.0, 0.5, 1.0], rho, np.asarray(prof.ne)) * density_factor,
            ])
            profile = np.concatenate([
                core,
                [density_factor, current_factor],
                ne11 * density_factor,
                te11 / density_factor,
            ])
            harmonics = np.array([
                function(harmonic * angle)
                for harmonic in range(1, 5)
                for angle in (pol_angle, tor_angle)
                for function in (np.sin, np.cos)
            ])
            wide = np.concatenate([profile, harmonics])
            if not (np.all(np.isfinite(article)) and np.all(np.isfinite(wide))):
                continue
            article_rows.append(article)
            wide_rows.append(wide)
            targets.append(target)
            frames.append(frame)
            points.append(point)

    return {
        "article": np.asarray(article_rows),
        "wide": np.asarray(wide_rows),
        "targets": np.asarray(targets),
        "frames": np.asarray(frames, dtype=np.int64),
        "points": np.asarray(points, dtype=np.int64),
        "total": np.array(n_frames * n_points),
        "n_frames": np.array(n_frames),
    }


def split_data(
    extracted: dict[int, dict[str, np.ndarray]], seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, list[int]]]]:
    rng = np.random.default_rng(seed)
    keys = ("article", "wide", "targets", "frames", "points", "shots")
    chunks = {
        split: {key: [] for key in keys}
        for split in ("train", "validation", "test")
    }
    assignments: dict[str, dict[str, list[int]]] = {}
    for shot, data in extracted.items():
        shuffled = rng.permutation(np.unique(data["frames"]))
        if len(shuffled) < 3:
            raise ValueError(f"Shot {shot} has fewer than three valid frames")
        n_test = max(1, int(round(0.1 * len(shuffled))))
        n_validation = max(1, int(round(0.1 * len(shuffled))))
        assignment = {
            "train": shuffled[:-(n_validation+n_test)].tolist(),
            "validation": shuffled[-(n_validation+n_test):-n_test].tolist(),
            "test": shuffled[-n_test:].tolist(),
        }
        assignments[str(shot)] = assignment
        for split, selected in assignment.items():
            mask = np.isin(data["frames"], selected)
            for key in ("article", "wide", "targets", "frames", "points"):
                chunks[split][key].append(data[key][mask])
            chunks[split]["shots"].append(np.full(mask.sum(), shot, dtype=np.int64))
    return {
        split: {key: np.concatenate(values) for key, values in split_chunks.items()}
        for split, split_chunks in chunks.items()
    }, assignments


def initialize(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class ArticleNet(nn.Module):
    """The article-compatible 14-60-60-60-5 dense ReLU network."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(14, 60), nn.ReLU(),
            nn.Linear(60, 60), nn.ReLU(),
            nn.Linear(60, 60), nn.ReLU(),
            nn.Linear(60, 5),
        )
        initialize(self)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class WideFourierProfileNet(nn.Module):
    """Shared 256-192-128 SiLU trunk and five independent 64-1 heads."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(51, 256), nn.SiLU(),
            nn.Linear(256, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
            for _ in OUTPUT_NAMES
        ])
        initialize(self)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(values)
        return torch.cat([head(shared) for head in self.heads], dim=1)


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(OUTPUT_NAMES):
        residual = prediction[:, index] - target[:, index]
        denominator = np.sum((target[:, index] - target[:, index].mean())**2)
        result[str(name)] = {
            "r2": float(1.0 - np.sum(residual**2) / denominator),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
        }
    return result


def train_model(
    name: str,
    model_factory: Callable[[], nn.Module],
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_mean = features["train"].mean(axis=0)
    x_scale = features["train"].std(axis=0)
    y_mean = targets["train"].mean(axis=0)
    y_scale = targets["train"].std(axis=0)
    x_scale[x_scale == 0.0] = 1.0
    if np.any(y_scale == 0.0):
        raise ValueError("A target has zero training variance")
    x_scaled = {split: (values-x_mean)/x_scale for split, values in features.items()}
    y_scaled = {split: (values-y_mean)/y_scale for split, values in targets.items()}

    model = model_factory()
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1.0e-5
        )
    loss_function = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(as_tensor(x_scaled["train"]), as_tensor(y_scaled["train"])),
        batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_x = as_tensor(x_scaled["validation"])
    validation_y = as_tensor(y_scaled["validation"])
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = {"train": [], "validation": []}
    print(f"\n{name}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    for epoch in range(1, config["epochs"]+1):
        model.train()
        total, elements = 0.0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_y.numel()
            elements += batch_y.numel()
        train_loss = total/elements
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x), validation_y).item())
        if scheduler is not None:
            scheduler.step(validation_loss)
        history["train"].append(train_loss)
        history["validation"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  epoch {epoch:3d}: train={train_loss:.6f}, "
                f"validation={validation_loss:.6f}, best={best_loss:.6f}", flush=True
            )
        if stale >= config["patience"]:
            break
    if best_state is None:
        raise RuntimeError(f"{name} produced no checkpoint")
    model.load_state_dict(best_state)
    predictions: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            scaled_prediction = np.asarray(model(as_tensor(x_scaled[split])).tolist(), dtype=np.float64)
            predictions[split] = scaled_prediction*y_scale+y_mean
    result = {
        "config": config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "stopped_epoch": len(history["train"]),
        "validation_scaled_mse": best_loss,
        "test_scaled_mse": float(np.mean(((predictions["test"]-y_mean)/y_scale-y_scaled["test"])**2)),
        "test_metrics": calculate_metrics(targets["test"], predictions["test"]),
        "history": history,
    }
    scalers = {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale}
    return model, predictions, scalers, result


def save_model(
    path: Path, model: nn.Module, scalers: dict[str, np.ndarray],
    input_names: np.ndarray, result: dict[str, Any],
) -> None:
    arrays: dict[str, np.ndarray] = {
        **scalers,
        "input_names": input_names,
        "output_names": OUTPUT_NAMES,
        "config_json": np.array(json.dumps(result["config"], sort_keys=True)),
    }
    for key, value in model.state_dict().items():
        arrays[f"state__{key}"] = np.asarray(value.detach().tolist(), dtype=np.float32)
    np.savez_compressed(path, **arrays)


def add_parity_row(
    axes: np.ndarray, targets: np.ndarray, predictions: np.ndarray,
    metrics: dict[str, dict[str, float]], ylabel: str,
) -> None:
    for index, (name, axis) in enumerate(zip(OUTPUT_NAMES, axes)):
        truth, estimate = targets[:, index], predictions[:, index]
        low = min(float(truth.min()), float(estimate.min()))
        high = max(float(truth.max()), float(estimate.max()))
        padding = 0.03*(high-low)
        axis.scatter(truth, estimate, s=8, alpha=0.42, edgecolors="none")
        axis.plot([low-padding, high+padding], [low-padding, high+padding], "r--", lw=1.2)
        axis.set(
            xlabel="TORBEAM", ylabel=ylabel, title=str(name),
            xlim=(low-padding, high+padding), ylim=(low-padding, high+padding),
        )
        value = metrics[str(name)]
        axis.text(
            0.04, 0.96,
            f"R² = {value['r2']:.4f}\nMAE = {value['mae']:.4g}\nRMSE = {value['rmse']:.4g}",
            transform=axis.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
        )
        axis.grid(alpha=0.2)


def make_figures(
    output_dir: Path, targets: np.ndarray, results: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, np.ndarray]],
) -> None:
    labels = {"article": "Article 60×3 ReLU", "wide_fourier": "Wide Fourier-profile NN"}
    for key in ("article", "wide_fourier"):
        figure, axes = plt.subplots(1, 5, figsize=(22, 4.4))
        add_parity_row(axes, targets, predictions[key]["test"], results[key]["test_metrics"], labels[key])
        figure.suptitle(f"{labels[key]} held-out-frame parity — mission 2513", fontsize=14)
        figure.tight_layout()
        figure.savefig(output_dir/f"{key}_r2.png", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(2, 5, figsize=(22, 8.8))
    for row, key in enumerate(("article", "wide_fourier")):
        add_parity_row(
            axes[row], targets, predictions[key]["test"], results[key]["test_metrics"], labels[key]
        )
        axes[row, 0].annotate(
            f"({chr(97+row)})", xy=(-0.22, 1.10), xycoords="axes fraction",
            fontsize=14, fontweight="bold",
        )
    figure.suptitle("Mission 2513 held-out R² comparison", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.3)
    figure.savefig(output_dir/"article_vs_wide_fourier_r2.png", dpi=180)
    plt.close(figure)

    x = np.arange(len(OUTPUT_NAMES)); width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5.8))
    for offset, key in zip((-width/2, width/2), ("article", "wide_fourier")):
        values = [results[key]["test_metrics"][str(name)]["r2"] for name in OUTPUT_NAMES]
        bars = axis.bar(x+offset, values, width, label=labels[key])
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    minimum = min(
        results[key]["test_metrics"][str(name)]["r2"]
        for key in results for name in OUTPUT_NAMES
    )
    axis.set(
        ylabel="Held-out R²", title="Mission 2513 R² comparison",
        xticks=x, xticklabels=OUTPUT_NAMES,
        ylim=(min(0.0, minimum-0.08), 1.04),
    )
    axis.grid(axis="y", alpha=0.25); axis.legend()
    figure.tight_layout(); figure.savefig(output_dir/"r2_bar_comparison.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, key in zip(axes, ("article", "wide_fourier")):
        history = results[key]["history"]
        epochs = np.arange(1, len(history["train"])+1)
        axis.semilogy(epochs, history["train"], label="training")
        axis.semilogy(epochs, history["validation"], label="validation")
        axis.axvline(results[key]["best_epoch"], color="k", ls="--", lw=1, label="selected epoch")
        axis.set(xlabel="Epoch", ylabel="Scaled MSE", title=labels[key]); axis.grid(alpha=0.25); axis.legend()
    figure.suptitle("Mission 2513 training histories")
    figure.tight_layout(); figure.savefig(output_dir/"training_histories.png", dpi=180); plt.close(figure)


def write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    rows = []
    for name in OUTPUT_NAMES:
        article = report["models"]["article"]["test_metrics"][str(name)]["r2"]
        wide = report["models"]["wide_fourier"]["test_metrics"][str(name)]["r2"]
        rows.append(f"| `{name}` | {article:.6f} | {wide:.6f} | {wide-article:+.6f} |")
    counts = report["sample_counts"]
    text = f"""# Mission 2513: article model versus wide Fourier-profile network

Both models were trained from scratch exclusively on the 11 complete shot pairs
in `../torbeam_training_data_mission2513`. They use the same equilibrium-frame
split (seed 42), with {counts['train']:,} training, {counts['validation']:,}
validation, and {counts['test']:,} untouched test samples. Scalers were fitted
on training samples only, and checkpoints were selected only by validation MSE.

Of 33,000 raw TORBEAM evaluations, 4,674 pass the established convergence and
physical-validity filters. Successful evaluations are unevenly distributed
between frames, so the whole-frame split produces a relatively small 268-sample
test partition. The comparison is controlled because both models use exactly
the same saved frames and samples, but the absolute R2 values should be regarded
as specific to this fixed split.

| Output | Article 60x3 ReLU R2 | Wide Fourier-profile R2 | Difference |
|---|---:|---:|---:|
{chr(10).join(rows)}

Aggregate scaled MSE:

| Model | Validation | Held-out test |
|---|---:|---:|
| Article 60x3 ReLU | {report['models']['article']['validation_scaled_mse']:.6f} | {report['models']['article']['test_scaled_mse']:.6f} |
| Wide Fourier-profile NN | {report['models']['wide_fourier']['validation_scaled_mse']:.6f} | {report['models']['wide_fourier']['test_scaled_mse']:.6f} |

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
"""
    (output_dir/"README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"torbeam_training_data_mission2513")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"mission2513_model_comparison")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))

    shots = discover_shots(data_dir)
    print(f"Mission 2513 only: {data_dir}", flush=True)
    print(f"Shots ({len(shots)}): {', '.join(map(str, shots))}", flush=True)
    extracted: dict[int, dict[str, np.ndarray]] = {}
    validity: dict[str, Any] = {}
    for shot in shots:
        data = extract_shot(data_dir, shot)
        extracted[shot] = data
        validity[str(shot)] = {
            "frames": int(data["n_frames"]), "total": int(data["total"]),
            "valid": len(data["targets"]),
            "valid_percent": 100.0*len(data["targets"])/int(data["total"]),
        }
        print(
            f"  {shot}: {len(data['targets'])}/{int(data['total'])} valid "
            f"across {int(data['n_frames'])} frames", flush=True
        )
    parts, assignments = split_data(extracted, seed=42)
    counts = {split: len(values["targets"]) for split, values in parts.items()}
    print(f"Split counts: {counts}", flush=True)

    features = {
        "article": {split: values["article"] for split, values in parts.items()},
        "wide_fourier": {split: values["wide"] for split, values in parts.items()},
    }
    targets = {split: values["targets"] for split, values in parts.items()}
    configurations = {
        "article": {
            "architecture": "14-60-60-60-5 ReLU", "optimizer": "Adam",
            "learning_rate": 6.3e-4, "batch_size": 128, "epochs": 1000,
            "patience": 250, "seed": 42,
        },
        "wide_fourier": {
            "architecture": "51-256-192-128 SiLU trunk; five 64-1 SiLU heads",
            "optimizer": "AdamW", "learning_rate": 3.5e-4, "weight_decay": 1.0e-5,
            "batch_size": 256, "epochs": 140, "patience": 35, "seed": 212,
        },
    }
    factories: dict[str, Callable[[], nn.Module]] = {
        "article": ArticleNet, "wide_fourier": WideFourierProfileNet,
    }
    model_objects: dict[str, nn.Module] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    scalers: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, dict[str, Any]] = {}
    for key in ("article", "wide_fourier"):
        model_objects[key], predictions[key], scalers[key], results[key] = train_model(
            key, factories[key], features[key], targets, configurations[key]
        )
        print(
            f"FINISHED {key}: validation={results[key]['validation_scaled_mse']:.6f}, "
            f"test={results[key]['test_scaled_mse']:.6f}", flush=True
        )

    save_model(output_dir/"article_model.npz", model_objects["article"], scalers["article"], ARTICLE_NAMES, results["article"])
    save_model(
        output_dir/"wide_fourier_profile_model.npz", model_objects["wide_fourier"],
        scalers["wide_fourier"], WIDE_NAMES, results["wide_fourier"],
    )
    cache_arrays: dict[str, np.ndarray] = {
        "article_names": ARTICLE_NAMES, "wide_names": WIDE_NAMES, "output_names": OUTPUT_NAMES,
    }
    prediction_arrays: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        cache_arrays[f"{split}_article"] = parts[split]["article"]
        cache_arrays[f"{split}_wide"] = parts[split]["wide"]
        cache_arrays[f"{split}_targets"] = parts[split]["targets"]
        cache_arrays[f"{split}_shots"] = parts[split]["shots"]
        cache_arrays[f"{split}_frames"] = parts[split]["frames"]
        prediction_arrays[f"{split}_targets"] = parts[split]["targets"]
        prediction_arrays[f"{split}_article_predictions"] = predictions["article"][split]
        prediction_arrays[f"{split}_wide_fourier_predictions"] = predictions["wide_fourier"][split]
    np.savez_compressed(output_dir/"dataset_cache.npz", **cache_arrays)
    np.savez_compressed(output_dir/"predictions.npz", **prediction_arrays)
    report = {
        "mission": 2513, "data_directory": str(data_dir), "shots": list(shots),
        "validity": validity, "sample_counts": counts, "split_seed": 42,
        "split_unit": "equilibrium frame within each shot", "frame_assignment": assignments,
        "feature_counts": {"article": 14, "wide_fourier": 51}, "models": results,
    }
    (output_dir/"training_report.json").write_text(json.dumps(report, indent=2)+"\n")
    make_figures(output_dir, targets["test"], results, predictions)
    write_readme(output_dir, report)

    print("\nHeld-out R2 comparison:", flush=True)
    for name in OUTPUT_NAMES:
        a = results["article"]["test_metrics"][str(name)]["r2"]
        w = results["wide_fourier"]["test_metrics"][str(name)]["r2"]
        print(f"  {name:7s}: article={a:.4f}, wide={w:.4f}, difference={w-a:+.4f}", flush=True)
    print(f"Artifacts written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
