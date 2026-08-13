#!/usr/bin/env python3
"""Run either deployable mission-2512 grouped exit-label model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch

from grouped_utils import GROUP_IDS, GROUP_NAMES, apply_non_normal_scale
from train_validate_grouped import GroupedMLP, aligned_tree_probabilities


HERE = Path(__file__).resolve().parent


def profile_transform(values: np.ndarray, checkpoint: dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    transformed = np.where(
        missing,
        np.asarray(checkpoint["imputer_statistics"])[None, :],
        values,
    )
    indicators = np.asarray(checkpoint["indicator_features"], dtype=int)
    if len(indicators):
        transformed = np.column_stack(
            [transformed, missing[:, indicators].astype(float)]
        )
    return (
        (transformed - np.asarray(checkpoint["scaler_mean"]))
        / np.asarray(checkpoint["scaler_scale"])
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "best"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"),
        help="Read <split>_compact/profile from the dataset cache",
    )
    parser.add_argument("--key", help="Explicit feature-matrix key in an external NPZ")
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
        bundle = joblib.load(HERE / "models/small_grouped_tree.joblib")
        transformed = bundle["imputer"].transform(values)
        probabilities = aligned_tree_probabilities(bundle["model"], transformed)
        scale = float(bundle["non_normal_probability_scale"])
    else:
        checkpoint = torch.load(
            HERE / "models/best_grouped_profile_mlp.pt", map_location="cpu"
        )
        transformed = profile_transform(values, checkpoint)
        model = GroupedMLP(
            transformed.shape[1], dropout=checkpoint["config"]["dropout"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(transformed), 2048):
                chunks.append(
                    torch.softmax(
                        model(torch.from_numpy(transformed[start:start + 2048])), dim=1
                    ).numpy()
                )
        probabilities = np.concatenate(chunks)
        scale = float(checkpoint["non_normal_probability_scale"])
    probabilities = apply_non_normal_scale(probabilities, scale)
    predicted = probabilities.argmax(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["predicted_group_id", "predicted_exit_category", "non_normal_probability"]
            + [f"probability_{name.replace(' ', '_').replace(',', '')}" for name in GROUP_NAMES]
        )
        for group_id, row in zip(predicted, probabilities):
            writer.writerow(
                [int(group_id), GROUP_NAMES[group_id], float(1.0 - row[0]), *map(float, row)]
            )
    print(f"Wrote {len(predicted)} grouped predictions to {args.output}")


if __name__ == "__main__":
    main()

