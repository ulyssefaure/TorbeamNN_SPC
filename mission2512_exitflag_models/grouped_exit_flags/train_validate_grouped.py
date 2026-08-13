#!/usr/bin/env python3
"""Train/select the two retained architectures using grouped exit flags.

Only training and validation arrays are accessed here. Test arrays are read
later by ``evaluate_grouped.py`` after every choice is locked and serialized.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from grouped_utils import (
    GROUP_IDS,
    GROUP_NAMES,
    GROUP_SOURCE_FLAGS,
    apply_non_normal_scale,
    best_non_normal_scale,
    class_sample_weights,
    evaluate_grouped,
    group_raw_flags,
)


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
MODELS = HERE / "models"
RESULTS = HERE / "results"
TREE_WEIGHT_POWERS = (0.0, 0.2, 0.35, 0.45, 0.55, 0.7, 0.85)
MLP_WEIGHT_POWERS = (0.25, 0.35, 0.45, 0.55, 0.65)
MAX_EPOCHS = 100
PATIENCE = 20
SEED = 42


class GroupedMLP(nn.Module):
    """Same retained 256-256-128 profile architecture, with five outputs."""

    def __init__(self, n_features: int, dropout: float = 0.12):
        super().__init__()
        widths = (256, 256, 128)
        layers: list[nn.Module] = []
        previous = n_features
        for width in widths:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, len(GROUP_IDS)))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def aligned_tree_probabilities(model: DecisionTreeClassifier, values: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(values)
    aligned = np.zeros((len(values), len(GROUP_IDS)), dtype=float)
    aligned[:, np.asarray(model.classes_, dtype=int)] = raw
    return aligned


def prepare_profile_features(
    train_values: np.ndarray,
    validation_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(train_values)).astype(np.float32)
    validation = scaler.transform(imputer.transform(validation_values)).astype(np.float32)
    return train, validation, imputer, scaler


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = np.load(PARENT / "dataset_cache.npz")
    # Deliberately access only train/validation keys in this script.
    y_train = group_raw_flags(data["train_labels"])
    y_validation = group_raw_flags(data["validation_labels"])

    selection_records: list[dict] = []

    # Small model: retain compact inputs and a depth-8, leaf>=5 decision tree.
    compact_imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )
    x_train_compact = compact_imputer.fit_transform(data["train_compact"])
    x_validation_compact = compact_imputer.transform(data["validation_compact"])
    best_tree_choice = None
    for power in TREE_WEIGHT_POWERS:
        started = time.perf_counter()
        tree = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            criterion="log_loss",
            random_state=SEED,
        )
        tree.fit(
            x_train_compact,
            y_train,
            sample_weight=class_sample_weights(y_train, power),
        )
        raw_probability = aligned_tree_probabilities(tree, x_validation_compact)
        scale, metrics, adjusted = best_non_normal_scale(
            raw_probability, y_validation
        )
        record = {
            "model": "small_tree",
            "class_weight_power": power,
            "non_normal_probability_scale": scale,
            "fit_seconds": time.perf_counter() - started,
            "node_count": int(tree.tree_.node_count),
            "depth": int(tree.tree_.max_depth),
            "validation": metrics,
        }
        selection_records.append(record)
        key = (
            metrics["grouped"]["macro_f1_observed"],
            metrics["grouped"]["accuracy"],
            -abs(power - 0.45),
        )
        if best_tree_choice is None or key > best_tree_choice[0]:
            best_tree_choice = (
                key, tree, power, scale, metrics, adjusted.copy(), record
            )
        print(
            f"tree power={power:.2f} scale={scale:.3f} "
            f"macroF1={metrics['grouped']['macro_f1_observed']:.5f} "
            f"accuracy={metrics['grouped']['accuracy']:.5f}",
            flush=True,
        )
    if best_tree_choice is None:
        raise RuntimeError("No tree candidate completed")
    _, tree, tree_power, tree_scale, tree_metrics, tree_probability, tree_record = best_tree_choice
    tree_bundle = {
        "model": tree,
        "imputer": compact_imputer,
        "feature_names": data["compact_names"],
        "group_ids": GROUP_IDS,
        "group_names": GROUP_NAMES,
        "group_source_flags": GROUP_SOURCE_FLAGS,
        "non_normal_probability_scale": tree_scale,
        "config": {
            "family": "decision_tree",
            "max_depth": 8,
            "min_samples_leaf": 5,
            "criterion": "log_loss",
            "class_weight_power": tree_power,
            "feature_set": "compact",
        },
        "training_scope": "mission-2512 training split only; validation selected weights/scale",
    }
    joblib.dump(tree_bundle, MODELS / "small_grouped_tree_evaluation.joblib", compress=3)
    np.savez_compressed(
        RESULTS / "small_validation_predictions.npz",
        true_group=y_validation,
        predicted_group=tree_probability.argmax(axis=1),
        probabilities=tree_probability,
    )

    # Best model: retain the 111-input profile tier and 256-256-128 MLP.
    x_train_profile, x_validation_profile, profile_imputer, profile_scaler = (
        prepare_profile_features(data["train_profile"], data["validation_profile"])
    )
    validation_tensor = torch.from_numpy(x_validation_profile)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train_profile), torch.from_numpy(y_train)
    )
    best_mlp_choice = None
    for power in MLP_WEIGHT_POWERS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.set_num_threads(4)
        loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        model = GroupedMLP(x_train_profile.shape[1], dropout=0.12)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=8.0e-4, weight_decay=2.0e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=5.0e-5
        )
        sample_weights = class_sample_weights(y_train, power)
        class_weights = np.asarray(
            [sample_weights[y_train == label][0] for label in GROUP_IDS],
            dtype=np.float32,
        )
        class_weights_tensor = torch.from_numpy(class_weights)
        run_best = None
        stale_epochs = 0
        started = time.perf_counter()
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for features, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = nn.functional.cross_entropy(
                    logits, targets, weight=class_weights_tensor
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                raw_probability = torch.softmax(model(validation_tensor), dim=1).numpy()
            scale, metrics, adjusted = best_non_normal_scale(
                raw_probability, y_validation
            )
            key = (
                metrics["grouped"]["macro_f1_observed"],
                metrics["grouped"]["accuracy"],
                -abs(scale - 1.0),
            )
            if run_best is None or key > run_best[0]:
                run_best = (
                    key,
                    epoch,
                    deepcopy(model.state_dict()),
                    scale,
                    metrics,
                    adjusted.copy(),
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch % 10 == 0:
                print(
                    f"MLP power={power:.2f} epoch={epoch} "
                    f"best_macroF1={run_best[0][0]:.5f}",
                    flush=True,
                )
            if epoch >= 25 and stale_epochs >= PATIENCE:
                break
        if run_best is None:
            raise RuntimeError(f"No MLP checkpoint for weight power {power}")
        key, epoch, state, scale, metrics, adjusted = run_best
        record = {
            "model": "profile_mlp",
            "class_weight_power": power,
            "non_normal_probability_scale": scale,
            "best_epoch": epoch,
            "fit_seconds": time.perf_counter() - started,
            "parameter_count": int(sum(value.numel() for value in model.parameters())),
            "validation": metrics,
        }
        selection_records.append(record)
        global_key = (*key, -abs(power - 0.5))
        if best_mlp_choice is None or global_key > best_mlp_choice[0]:
            best_mlp_choice = (
                global_key, power, epoch, state, scale, metrics, adjusted, record
            )
        print(
            f"MLP power={power:.2f} selected epoch={epoch} scale={scale:.3f} "
            f"macroF1={metrics['grouped']['macro_f1_observed']:.5f} "
            f"accuracy={metrics['grouped']['accuracy']:.5f}",
            flush=True,
        )
    if best_mlp_choice is None:
        raise RuntimeError("No MLP candidate completed")
    _, mlp_power, mlp_epoch, mlp_state, mlp_scale, mlp_metrics, mlp_probability, mlp_record = best_mlp_choice
    mlp_checkpoint = {
        "state_dict": mlp_state,
        "imputer_statistics": profile_imputer.statistics_,
        "indicator_features": profile_imputer.indicator_.features_,
        "scaler_mean": profile_scaler.mean_,
        "scaler_scale": profile_scaler.scale_,
        "feature_names": data["profile_names"],
        "group_ids": GROUP_IDS,
        "group_names": GROUP_NAMES,
        "group_source_flags": GROUP_SOURCE_FLAGS,
        "non_normal_probability_scale": mlp_scale,
        "config": {
            "family": "neural_mlp",
            "hidden": [256, 256, 128],
            "dropout": 0.12,
            "output_count": 5,
            "loss": "weighted_cross_entropy",
            "class_weight_power": mlp_power,
            "epochs": mlp_epoch,
            "feature_set": "profile",
        },
        "training_scope": "mission-2512 training split only; validation selected weights/epoch/scale",
    }
    torch.save(mlp_checkpoint, MODELS / "best_grouped_profile_mlp_evaluation.pt")
    np.savez_compressed(
        RESULTS / "best_validation_predictions.npz",
        true_group=y_validation,
        predicted_group=mlp_probability.argmax(axis=1),
        probabilities=mlp_probability,
    )

    selection = {
        "target_mapping": {
            GROUP_NAMES[group_id]: raw_flags
            for group_id, raw_flags in GROUP_SOURCE_FLAGS.items()
        },
        "selection_metric": "five-class macro F1 on the validation frames",
        "test_access_policy": (
            "This script never accesses a test_* array. The chosen configurations "
            "are serialized before evaluate_grouped.py reads the test split."
        ),
        "small_tree": tree_record,
        "best_profile_mlp": mlp_record,
        "all_validation_candidates": selection_records,
    }
    (RESULTS / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("\nLocked selections before test access:")
    print(json.dumps({
        "small_tree": tree_record,
        "best_profile_mlp": mlp_record,
    }, indent=2))


if __name__ == "__main__":
    main()

