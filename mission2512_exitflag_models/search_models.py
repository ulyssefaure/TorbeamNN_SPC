#!/usr/bin/env python3
"""Validation-only model search for mission-2512 exit flags.

This script deliberately never reads the held-out test arrays.  Candidate
selection is based on macro F1 over exit codes represented in validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from metrics_utils import class_sample_weights, encode_flags, evaluate_predictions


HERE = Path(__file__).resolve().parent
CACHE = HERE / "dataset_cache.npz"
RESULTS = HERE / "validation_search.jsonl"
PREDICTIONS = HERE / "validation_probabilities"
GEOMETRY_NAMES = {
    "launch_x", "launch_y", "launch_z", "waist_vertical", "waist_horizontal",
    "curvature_vertical", "curvature_horizontal", "launcher_geometry_missing",
}


def feature_variant(data: np.lib.npyio.NpzFile, split: str, variant: str) -> np.ndarray:
    base = variant.split("_")[0]
    values = data[f"{split}_{base}"]
    names = data[f"{base}_names"].astype(str)
    keep = np.ones(len(names), dtype=bool)
    if variant.endswith("_notime") or variant.endswith("_core"):
        keep &= names != "frame_time_s"
    if variant.endswith("_core"):
        keep &= ~np.isin(names, list(GEOMETRY_NAMES))
    return values[:, keep]


def candidate_id(config: dict) -> str:
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    return f"{config['family']}_{config['variant']}_{digest}"


def configurations(stage: str) -> list[dict]:
    configs: list[dict] = []
    if stage in {"small", "all"}:
        configs.append({"family": "dummy", "variant": "compact", "params": {}, "weight_power": 0.0})
        for variant in ("compact", "profile"):
            for depth in (3, 4, 5, 6, 7, 8):
                for leaf in (5, 20):
                    for power in (0.0, 0.45, 0.8):
                        configs.append(
                            {
                                "family": "decision_tree", "variant": variant,
                                "params": {"max_depth": depth, "min_samples_leaf": leaf, "criterion": "log_loss"},
                                "weight_power": power,
                            }
                        )
        for variant in ("compact", "profile"):
            for c_value in (0.03, 0.1, 0.3, 1.0, 3.0):
                for power in (0.0, 0.5, 0.8):
                    configs.append(
                        {
                            "family": "logistic", "variant": variant,
                            "params": {"C": c_value}, "weight_power": power,
                        }
                    )
    if stage in {"lgb", "all"}:
        # Feature and class-weight probes.
        for variant in ("compact", "profile", "rich"):
            for power in (0.0, 0.5, 0.8):
                configs.append(
                    {
                        "family": "lightgbm", "variant": variant,
                        "params": {
                            "n_estimators": 280, "learning_rate": 0.06,
                            "num_leaves": 63, "min_child_samples": 20,
                            "colsample_bytree": 0.85, "subsample": 0.9,
                            "reg_lambda": 1.0,
                        },
                        "weight_power": power,
                    }
                )
        # Capacity/regularization probes on the two information-rich tiers.
        for variant in ("profile", "rich"):
            for leaves, child, regularization in (
                (31, 10, 0.3), (63, 40, 3.0), (127, 12, 2.0),
            ):
                for power in (0.4, 0.7):
                    configs.append(
                        {
                            "family": "lightgbm", "variant": variant,
                            "params": {
                                "n_estimators": 450, "learning_rate": 0.04,
                                "num_leaves": leaves, "min_child_samples": child,
                                "colsample_bytree": 0.9, "subsample": 0.9,
                                "reg_lambda": regularization,
                            },
                            "weight_power": power,
                        }
                    )
    if stage in {"forest", "all"}:
        forest_probes = (
            ("compact", "sqrt", 1, 0.0),
            ("compact", 0.5, 2, 0.5),
            ("profile", 0.3, 1, 0.0),
            ("profile", 0.3, 1, 0.5),
            ("profile", 0.5, 2, 0.5),
            ("profile", 0.5, 2, 0.8),
            ("profile", 0.8, 3, 0.5),
            ("rich", 0.25, 1, 0.0),
            ("rich", 0.25, 1, 0.5),
            ("rich", 0.5, 2, 0.5),
            ("rich", 0.5, 2, 0.8),
            ("rich", 0.8, 3, 0.5),
        )
        for variant, max_features, leaf, power in forest_probes:
            configs.append(
                {
                    "family": "extra_trees", "variant": variant,
                    "params": {
                        "n_estimators": 450, "max_features": max_features,
                        "min_samples_leaf": leaf,
                    },
                    "weight_power": power,
                }
            )
    if stage in {"hist", "all"}:
        for variant in ("profile", "rich"):
            for leaves, leaf, l2 in ((31, 20, 1.0), (63, 20, 3.0)):
                for power in (0.0, 0.55):
                    configs.append(
                        {
                            "family": "hist_gradient", "variant": variant,
                            "params": {
                                "max_iter": 350, "learning_rate": 0.08,
                                "max_leaf_nodes": leaves, "min_samples_leaf": leaf,
                                "l2_regularization": l2, "early_stopping": False,
                            },
                            "weight_power": power,
                        }
                    )
    if stage in {"xgb", "all"}:
        for variant in ("profile", "rich"):
            for depth, child, regularization in ((4, 2, 1.0), (7, 6, 3.0)):
                for power in (0.0, 0.55):
                    configs.append(
                        {
                            "family": "xgboost", "variant": variant,
                            "params": {
                                "n_estimators": 500, "learning_rate": 0.05,
                                "max_depth": depth, "min_child_weight": child,
                                "subsample": 0.9, "colsample_bytree": 0.85,
                                "reg_lambda": regularization,
                            },
                            "weight_power": power,
                        }
                    )
    return configs


def fit_and_predict(
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    family = config["family"]
    params = config["params"]
    weights = class_sample_weights(y_train, config["weight_power"])
    extra: dict[str, int | float] = {}
    if family == "dummy":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x_train, y_train)
    elif family in {"decision_tree", "extra_trees", "logistic"}:
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        x_train = imputer.fit_transform(x_train)
        x_val = imputer.transform(x_val)
        if family == "decision_tree":
            model = DecisionTreeClassifier(**params, random_state=42)
        elif family == "extra_trees":
            model = ExtraTreesClassifier(**params, random_state=42, n_jobs=4)
        else:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)
            model = LogisticRegression(**params, max_iter=1000, solver="lbfgs", tol=1e-6)
        model.fit(x_train, y_train, sample_weight=weights)
        if family == "decision_tree":
            extra.update(node_count=int(model.tree_.node_count), depth=int(model.tree_.max_depth))
    elif family == "lightgbm":
        model = LGBMClassifier(
            objective="multiclass", num_class=8, **params,
            random_state=42, n_jobs=4, verbosity=-1,
        )
        model.fit(
            x_train, y_train, sample_weight=weights,
            eval_set=[(x_val, y_val)], eval_metric="multi_logloss",
            callbacks=[early_stopping(35, verbose=False)],
        )
        extra["best_iteration"] = int(model.best_iteration_)
    elif family == "hist_gradient":
        model = HistGradientBoostingClassifier(**params, random_state=42)
        model.fit(x_train, y_train, sample_weight=weights)
    elif family == "xgboost":
        model = XGBClassifier(
            objective="multi:softprob", num_class=8, eval_metric="mlogloss",
            tree_method="hist", **params, random_state=42, n_jobs=4,
        )
        model.fit(x_train, y_train, sample_weight=weights, verbose=False)
    else:
        raise ValueError(f"Unknown family {family}")
    probability = model.predict_proba(x_val)
    # All eight classes occur in training, but keep alignment explicit.
    aligned = np.zeros((len(x_val), 8), dtype=float)
    aligned[:, np.asarray(model.classes_, dtype=int)] = probability
    return aligned, extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("small", "lgb", "forest", "hist", "xgb", "all"), required=True)
    parser.add_argument("--reset", action="store_true", help="Clear previous search records before this stage")
    args = parser.parse_args()
    if args.reset:
        RESULTS.unlink(missing_ok=True)
        if PREDICTIONS.exists():
            for path in PREDICTIONS.glob("*.npz"):
                path.unlink()
    PREDICTIONS.mkdir(exist_ok=True)
    completed: set[str] = set()
    if RESULTS.exists():
        completed = {json.loads(line)["id"] for line in RESULTS.read_text().splitlines() if line.strip()}

    data = np.load(CACHE)
    y_train = encode_flags(data["train_labels"])
    y_val = encode_flags(data["validation_labels"])
    configs = configurations(args.stage)
    for index, config in enumerate(configs, 1):
        identity = candidate_id(config)
        if identity in completed:
            continue
        x_train = feature_variant(data, "train", config["variant"])
        x_val = feature_variant(data, "validation", config["variant"])
        started = time.perf_counter()
        try:
            probability, extra = fit_and_predict(config, x_train, y_train, x_val, y_val)
            predicted = probability.argmax(axis=1)
            metrics = evaluate_predictions(y_val, predicted, probability)
            elapsed = time.perf_counter() - started
            record = {
                "id": identity, "status": "ok", "config": config,
                "feature_count": int(x_train.shape[1]), "fit_seconds": elapsed,
                "model_details": extra, "validation": metrics,
            }
            np.savez_compressed(PREDICTIONS / f"{identity}.npz", probabilities=probability)
            score = metrics["exact"]["macro_f1_observed"]
            accuracy = metrics["exact"]["accuracy"]
            print(
                f"[{index:03d}/{len(configs):03d}] {identity} "
                f"macroF1={score:.5f} accuracy={accuracy:.5f} time={elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:  # preserve the rest of a long search
            record = {
                "id": identity, "status": "error", "config": config,
                "error": f"{type(exc).__name__}: {exc}",
                "fit_seconds": time.perf_counter() - started,
            }
            print(f"[{index:03d}/{len(configs):03d}] {identity} ERROR {record['error']}", flush=True)
        with RESULTS.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    records = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    valid = [record for record in records if record["status"] == "ok"]
    valid.sort(key=lambda item: item["validation"]["exact"]["macro_f1_observed"], reverse=True)
    print("\nTop validation candidates:")
    for record in valid[:15]:
        exact = record["validation"]["exact"]
        binary = record["validation"]["binary_failure"]
        print(
            f"{record['id']}: macroF1={exact['macro_f1_observed']:.5f}, "
            f"accuracy={exact['accuracy']:.5f}, failure_AP={binary['pr_auc']:.5f}"
        )


if __name__ == "__main__":
    main()
