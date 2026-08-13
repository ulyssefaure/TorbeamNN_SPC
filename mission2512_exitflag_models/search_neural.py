#!/usr/bin/env python3
"""Validation-only neural-network probes for mission-2512 exit flags."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from metrics_utils import class_sample_weights, encode_flags, evaluate_predictions


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "validation_search.jsonl"
PROBABILITIES = HERE / "validation_probabilities"
CHECKPOINTS = HERE / "neural_search_checkpoints"


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_features
        for width in hidden:
            layers.extend(
                [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(dropout)]
            )
            previous = width
        layers.append(nn.Linear(previous, 8))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    raw = nn.functional.cross_entropy(logits, target, weight=class_weights, reduction="none")
    probability = torch.softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
    return (((1.0 - probability) ** gamma) * raw).mean()


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    PROBABILITIES.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    data = np.load(HERE / "dataset_cache.npz")
    y_train = encode_flags(data["train_labels"])
    y_val = encode_flags(data["validation_labels"])
    experiments = [
        {"variant": "profile", "hidden": [128, 64], "dropout": 0.08, "weight_power": 0.5, "gamma": 0.0},
        {"variant": "profile", "hidden": [256, 256, 128], "dropout": 0.12, "weight_power": 0.5, "gamma": 0.0},
        {"variant": "profile", "hidden": [320, 256, 128], "dropout": 0.10, "weight_power": 0.45, "gamma": 1.2},
        {"variant": "rich", "hidden": [256, 128], "dropout": 0.12, "weight_power": 0.5, "gamma": 0.0},
    ]
    completed = set()
    if RESULTS.exists():
        completed = {json.loads(line)["id"] for line in RESULTS.read_text().splitlines() if line.strip()}

    for experiment in experiments:
        config = {
            "family": "neural_mlp", "variant": experiment["variant"],
            "params": {
                "hidden": experiment["hidden"], "dropout": experiment["dropout"],
                "gamma": experiment["gamma"], "batch_size": 512,
                "learning_rate": 8.0e-4, "weight_decay": 2.0e-4,
            },
            "weight_power": experiment["weight_power"],
        }
        digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        identity = f"neural_mlp_{experiment['variant']}_{digest}"
        if identity in completed:
            continue
        started = time.perf_counter()
        x_train_raw = data[f"train_{experiment['variant']}"]
        x_val_raw = data[f"validation_{experiment['variant']}"]
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw)).astype(np.float32)
        x_val = scaler.transform(imputer.transform(x_val_raw)).astype(np.float32)
        train_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader = DataLoader(train_set, batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42))
        model = MLP(x_train.shape[1], experiment["hidden"], experiment["dropout"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90, eta_min=5.0e-5)
        per_sample = class_sample_weights(y_train, experiment["weight_power"])
        class_weights = np.zeros(8, dtype=np.float32)
        for label in range(8):
            class_weights[label] = float(per_sample[y_train == label][0])
        class_weights_tensor = torch.from_numpy(class_weights)
        val_tensor = torch.from_numpy(x_val)
        best_score = -np.inf
        best_epoch = 0
        best_state = None
        best_probability = None
        patience = 0
        for epoch in range(1, 121):
            model.train()
            for features, target in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                if experiment["gamma"] > 0:
                    loss = focal_cross_entropy(
                        logits, target, class_weights_tensor, experiment["gamma"]
                    )
                else:
                    loss = nn.functional.cross_entropy(logits, target, weight=class_weights_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                probability = torch.softmax(model(val_tensor), dim=1).numpy()
            metrics = evaluate_predictions(y_val, probability.argmax(axis=1), probability)
            score = metrics["exact"]["macro_f1_observed"]
            if score > best_score + 1.0e-5:
                best_score = score
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                best_probability = probability.copy()
                patience = 0
            else:
                patience += 1
            if epoch % 10 == 0:
                print(f"{identity} epoch={epoch} best_macroF1={best_score:.5f}", flush=True)
            if epoch >= 30 and patience >= 18:
                break
        if best_state is None or best_probability is None:
            raise RuntimeError("No neural checkpoint selected")
        model.load_state_dict(best_state)
        metrics = evaluate_predictions(y_val, best_probability.argmax(axis=1), best_probability)
        parameter_count = sum(value.numel() for value in model.parameters())
        checkpoint = {
            "config": config, "state_dict": best_state,
            "imputer_statistics": imputer.statistics_,
            "indicator_features": getattr(imputer.indicator_, "features_", np.empty(0, dtype=int)),
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "feature_names": data[f"{experiment['variant']}_names"],
        }
        torch.save(checkpoint, CHECKPOINTS / f"{identity}.pt")
        np.savez_compressed(PROBABILITIES / f"{identity}.npz", probabilities=best_probability)
        record = {
            "id": identity, "status": "ok", "config": config,
            "feature_count": int(x_train_raw.shape[1]),
            "fit_seconds": time.perf_counter() - started,
            "model_details": {
                "parameter_count": int(parameter_count), "best_epoch": best_epoch,
            },
            "validation": metrics,
        }
        with RESULTS.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"{identity} DONE macroF1={best_score:.5f} "
            f"accuracy={metrics['exact']['accuracy']:.5f} epoch={best_epoch}",
            flush=True,
        )


if __name__ == "__main__":
    main()

