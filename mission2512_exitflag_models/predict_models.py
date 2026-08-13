#!/usr/bin/env python3
"""Run either saved mission-2512 exit-flag classifier on feature matrices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch

from metrics_utils import CLASS_VALUES
from search_neural import MLP


HERE = Path(__file__).resolve().parent


def small_probabilities(bundle: dict, values: np.ndarray) -> np.ndarray:
    raw = bundle["model"].predict_proba(bundle["imputer"].transform(values))
    aligned = np.zeros((len(values), len(CLASS_VALUES)), dtype=float)
    aligned[:, np.asarray(bundle["model"].classes_, dtype=int)] = raw
    return aligned


def neural_probabilities(values: np.ndarray, checkpoint: dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    statistics = np.asarray(checkpoint["imputer_statistics"], dtype=float)
    transformed = np.where(missing, statistics[None, :], values)
    indicator_features = np.asarray(checkpoint["indicator_features"], dtype=int)
    if len(indicator_features):
        transformed = np.column_stack(
            [transformed, missing[:, indicator_features].astype(float)]
        )
    transformed = (
        (transformed - np.asarray(checkpoint["scaler_mean"]))
        / np.asarray(checkpoint["scaler_scale"])
    ).astype(np.float32)
    config = checkpoint["config"]["params"]
    model = MLP(transformed.shape[1], config["hidden"], config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(transformed), 2048):
            chunks.append(
                torch.softmax(model(torch.from_numpy(transformed[start:start + 2048])), dim=1).numpy()
            )
    return np.concatenate(chunks)


def apply_failure_scale(probability: np.ndarray, scale: float) -> np.ndarray:
    calibrated = np.asarray(probability, dtype=float).copy()
    calibrated[:, 1:] *= scale
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "best"), required=True)
    parser.add_argument("--input", type=Path, default=HERE / "dataset_cache.npz")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"),
        help="Read <split>_compact/profile from the dataset cache",
    )
    parser.add_argument("--key", help="Explicit matrix key for an external NPZ file")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.split is None) == (args.key is None):
        parser.error("Specify exactly one of --split or --key")
    archive = np.load(args.input)
    feature_set = "compact" if args.model == "small" else "profile"
    key = args.key or f"{args.split}_{feature_set}"
    if key not in archive:
        raise KeyError(f"{args.input} has no array named {key!r}")
    values = archive[key]
    if args.model == "small":
        bundle = joblib.load(HERE / "models/small_decision_tree.joblib")
        probability = small_probabilities(bundle, values)
    else:
        checkpoint = torch.load(HERE / "models/best_profile_mlp.pt", map_location="cpu")
        probability = neural_probabilities(values, checkpoint)
        probability = apply_failure_scale(
            probability, float(checkpoint["failure_probability_scale"])
        )
    predicted = CLASS_VALUES[probability.argmax(axis=1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["predicted_exit_flag", "failure_probability"]
            + [f"probability_flag_{int(flag)}" for flag in CLASS_VALUES]
        )
        for flag, row in zip(predicted, probability):
            writer.writerow([int(flag), float(1.0 - row[0]), *map(float, row)])
    print(f"Wrote {len(predicted)} predictions to {args.output}")


if __name__ == "__main__":
    main()
