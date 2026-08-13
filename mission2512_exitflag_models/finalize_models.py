#!/usr/bin/env python3
"""Fit the selected compact model and evaluate both selected models once.

Selection was performed exclusively on ``train`` and ``validation``.  This
script is the only code path that reads ``test_*`` arrays.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from metrics_utils import (
    CLASS_VALUES,
    FLAG_NAMES,
    class_sample_weights,
    encode_flags,
    evaluate_predictions,
)
from search_neural import MLP


HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
FIGURES = HERE / "figures"
RESULTS = HERE / "results"
SEARCH_CHECKPOINT = HERE / "neural_search_checkpoints/neural_mlp_profile_c22f0f34aa.pt"
BEST_FAILURE_SCALE = 0.55
SMALL_CONFIG = {
    "family": "decision_tree", "feature_set": "compact", "max_depth": 8,
    "min_samples_leaf": 5, "criterion": "log_loss", "class_weight_power": 0.45,
}
BEST_CONFIG = {
    "family": "neural_mlp", "feature_set": "profile", "hidden": [256, 256, 128],
    "dropout": 0.12, "loss": "weighted_cross_entropy", "class_weight_power": 0.5,
    "epochs": 10, "failure_probability_scale": BEST_FAILURE_SCALE,
}


def transform_from_checkpoint(values: np.ndarray, checkpoint: dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    statistics = np.asarray(checkpoint["imputer_statistics"], dtype=float)
    missing = np.isnan(values)
    transformed = np.where(missing, statistics[None, :], values)
    indicator_features = np.asarray(checkpoint["indicator_features"], dtype=int)
    if len(indicator_features):
        transformed = np.column_stack([transformed, missing[:, indicator_features].astype(float)])
    mean = np.asarray(checkpoint["scaler_mean"], dtype=float)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=float)
    return ((transformed - mean) / scale).astype(np.float32)


def neural_probabilities(values: np.ndarray, checkpoint: dict) -> np.ndarray:
    transformed = transform_from_checkpoint(values, checkpoint)
    config = checkpoint["config"]
    model = MLP(transformed.shape[1], config["params"]["hidden"], config["params"]["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(transformed), 2048):
            logits = model(torch.from_numpy(transformed[start:start + 2048]))
            chunks.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(chunks)


def apply_failure_scale(probability: np.ndarray, scale: float) -> np.ndarray:
    calibrated = np.asarray(probability, dtype=float).copy()
    calibrated[:, 1:] *= scale
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def fit_small(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    transformed = imputer.fit_transform(x_train)
    tree = DecisionTreeClassifier(
        max_depth=SMALL_CONFIG["max_depth"],
        min_samples_leaf=SMALL_CONFIG["min_samples_leaf"],
        criterion=SMALL_CONFIG["criterion"],
        random_state=42,
    )
    tree.fit(
        transformed, y_train,
        sample_weight=class_sample_weights(y_train, SMALL_CONFIG["class_weight_power"]),
    )
    return {"imputer": imputer, "model": tree}


def small_probabilities(bundle: dict, values: np.ndarray) -> np.ndarray:
    probability = bundle["model"].predict_proba(bundle["imputer"].transform(values))
    aligned = np.zeros((len(values), len(CLASS_VALUES)), dtype=float)
    aligned[:, np.asarray(bundle["model"].classes_, dtype=int)] = probability
    return aligned


def fit_full_development_neural(x_values: np.ndarray, y_values: np.ndarray) -> dict:
    """Refit the locked neural architecture on train+validation for deployment."""
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(x_values)).astype(np.float32)
    model = MLP(transformed.shape[1], BEST_CONFIG["hidden"], BEST_CONFIG["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-4)
    sample_weights = class_sample_weights(y_values, BEST_CONFIG["class_weight_power"])
    class_weights = np.zeros(8, dtype=np.float32)
    for label in range(8):
        class_weights[label] = float(sample_weights[y_values == label][0])
    class_weights_tensor = torch.from_numpy(class_weights)
    dataset = TensorDataset(torch.from_numpy(transformed), torch.from_numpy(y_values))
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(42))
    for _ in range(BEST_CONFIG["epochs"]):
        model.train()
        for features, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features), target, weight=class_weights_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return {
        "config": {
            "family": "neural_mlp", "variant": "profile",
            "params": {"hidden": BEST_CONFIG["hidden"], "dropout": BEST_CONFIG["dropout"]},
        },
        "state_dict": model.state_dict(),
        "imputer_statistics": imputer.statistics_,
        "indicator_features": imputer.indicator_.features_,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "failure_probability_scale": BEST_FAILURE_SCALE,
        "training_scope": "mission-2512 train+validation; held-out test excluded",
    }


def cluster_bootstrap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    repetitions: int = 2000,
) -> dict:
    rng = np.random.default_rng(20260811)
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    values = {name: [] for name in ("exact_accuracy", "macro_f1_observed", "failure_f1", "failure_pr_auc")}
    for _ in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        metrics = evaluate_predictions(y_true[chosen], probabilities[chosen].argmax(axis=1), probabilities[chosen])
        values["exact_accuracy"].append(metrics["exact"]["accuracy"])
        values["macro_f1_observed"].append(metrics["exact"]["macro_f1_observed"])
        values["failure_f1"].append(metrics["binary_failure"]["f1"])
        values["failure_pr_auc"].append(metrics["binary_failure"]["pr_auc"])
    return {
        name: {
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def plot_confusion(metrics: dict, title: str, path_stem: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix), where=support != 0)
    fig, axis = plt.subplots(figsize=(8.4, 7.1))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(8):
        for column in range(8):
            count = int(matrix[row, column])
            if count:
                axis.text(
                    column, row, str(count), ha="center", va="center", fontsize=8,
                    color="white" if normalized[row, column] > 0.55 else "black",
                )
    labels = [str(int(value)) for value in CLASS_VALUES]
    axis.set_xticks(range(8), labels)
    axis.set_yticks(range(8), labels)
    axis.set_xlabel("Predicted exit flag")
    axis.set_ylabel("True exit flag")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Fraction within true class")
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=220)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def save_figures(all_metrics: dict, y_test: np.ndarray, probabilities: dict) -> None:
    plot_confusion(all_metrics["small"]["test"], "Small decision tree — held-out test", FIGURES / "confusion_small")
    plot_confusion(all_metrics["best"]["test"], "Best profile MLP — held-out test", FIGURES / "confusion_best")

    names = ["Majority baseline", "Small tree", "Best MLP"]
    keys = ["baseline", "small", "best"]
    metric_specs = [
        ("Exact accuracy", lambda item: item["exact"]["accuracy"]),
        ("Macro F1", lambda item: item["exact"]["macro_f1_observed"]),
        ("Weighted F1", lambda item: item["exact"]["weighted_f1"]),
        ("Failure PR-AUC", lambda item: item["binary_failure"]["pr_auc"]),
    ]
    x = np.arange(len(metric_specs)); width = 0.24
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    for offset, (name, key) in enumerate(zip(names, keys)):
        values = [extractor(all_metrics[key]["test"]) for _, extractor in metric_specs]
        bars = axis.bar(x + (offset - 1) * width, values, width, label=name)
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axis.set_xticks(x, [label for label, _ in metric_specs])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Mission 2512 exit-flag models — held-out frames")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_model_comparison.png", dpi=220)
    fig.savefig(FIGURES / "test_model_comparison.pdf")
    plt.close(fig)

    x = np.arange(8)
    fig, axis = plt.subplots(figsize=(10, 5.2))
    for offset, (name, key) in enumerate((("Small tree", "small"), ("Best MLP", "best"))):
        values = [all_metrics[key]["test"]["per_class"][str(int(flag))]["f1"] for flag in CLASS_VALUES]
        bars = axis.bar(x + (offset - 0.5) * 0.36, values, 0.36, label=name)
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    supports = [all_metrics["best"]["test"]["per_class"][str(int(flag))]["support"] for flag in CLASS_VALUES]
    axis.set_xticks(x, [f"{int(flag)}\n(n={support})" for flag, support in zip(CLASS_VALUES, supports)])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("F1 score")
    axis.set_xlabel("Exit flag (held-out support)")
    axis.set_title("Per-class held-out performance")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_per_class_f1.png", dpi=220)
    fig.savefig(FIGURES / "test_per_class_f1.pdf")
    plt.close(fig)

    true_failure = y_test != 0
    fig, axis = plt.subplots(figsize=(6.8, 5.4))
    for label, key in (("Small tree", "small"), ("Best MLP", "best")):
        precision, recall, _ = precision_recall_curve(true_failure, 1.0 - probabilities[key][:, 0])
        ap = all_metrics[key]["test"]["binary_failure"]["pr_auc"]
        axis.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    prevalence = true_failure.mean()
    axis.axhline(prevalence, color="0.45", linestyle="--", label=f"Prevalence ({prevalence:.3f})")
    axis.set(xlabel="Failure recall", ylabel="Failure precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.set_title("Normal exit versus nonzero exit flag")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "test_failure_precision_recall.png", dpi=220)
    fig.savefig(FIGURES / "test_failure_precision_recall.pdf")
    plt.close(fig)

    counts = np.array([26254, 134, 544, 5555, 1, 51, 361, 100])
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.bar([str(int(flag)) for flag in CLASS_VALUES], counts)
    axis.set_yscale("log")
    axis.set_xlabel("Exit flag")
    axis.set_ylabel("Mission-2512 count (log scale)")
    axis.set_title("Severe exit-flag class imbalance")
    axis.bar_label(bars, labels=[f"{value:,}" for value in counts], fontsize=8, padding=2)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=220)
    fig.savefig(FIGURES / "class_distribution.pdf")
    plt.close(fig)


def main() -> None:
    for directory in (MODELS, FIGURES, RESULTS):
        directory.mkdir(exist_ok=True)
    data = np.load(HERE / "dataset_cache.npz")
    y_train = encode_flags(data["train_labels"])
    y_validation = encode_flags(data["validation_labels"])

    # Model selection is locked here, before any test array is accessed.
    small = fit_small(data["train_compact"], y_train)
    small_validation_probability = small_probabilities(small, data["validation_compact"])
    checkpoint = torch.load(SEARCH_CHECKPOINT, map_location="cpu")
    best_validation_probability = apply_failure_scale(
        neural_probabilities(data["validation_profile"], checkpoint), BEST_FAILURE_SCALE
    )
    validation_metrics = {
        "small": evaluate_predictions(
            y_validation, small_validation_probability.argmax(axis=1), small_validation_probability
        ),
        "best": evaluate_predictions(
            y_validation, best_validation_probability.argmax(axis=1), best_validation_probability
        ),
    }

    # First and only access to the held-out test arrays.
    y_test = encode_flags(data["test_labels"])
    small_test_probability = small_probabilities(small, data["test_compact"])
    best_test_probability = apply_failure_scale(
        neural_probabilities(data["test_profile"], checkpoint), BEST_FAILURE_SCALE
    )
    baseline_probability = np.zeros((len(y_test), 8), dtype=float)
    baseline_probability[:, 0] = 1.0
    probabilities = {
        "baseline": baseline_probability,
        "small": small_test_probability,
        "best": best_test_probability,
    }
    all_metrics = {
        "baseline": {
            "test": evaluate_predictions(y_test, baseline_probability.argmax(axis=1), baseline_probability)
        },
        "small": {
            "validation": validation_metrics["small"],
            "test": evaluate_predictions(y_test, small_test_probability.argmax(axis=1), small_test_probability),
        },
        "best": {
            "validation": validation_metrics["best"],
            "test": evaluate_predictions(y_test, best_test_probability.argmax(axis=1), best_test_probability),
        },
    }
    groups = np.asarray([f"{shot}:{frame}" for shot, frame in zip(data["test_shots"], data["test_frames"])])
    for key in ("small", "best"):
        all_metrics[key]["cluster_bootstrap_95_ci"] = cluster_bootstrap(
            y_test, probabilities[key], groups
        )
        per_shot = {}
        for shot in np.unique(data["test_shots"]):
            mask = data["test_shots"] == shot
            per_shot[str(int(shot))] = evaluate_predictions(
                y_test[mask], probabilities[key][mask].argmax(axis=1), probabilities[key][mask]
            )
        all_metrics[key]["per_shot_test"] = per_shot

    selection = {
        "selection_metric": "macro F1 over exit codes present in validation",
        "small": SMALL_CONFIG,
        "best": BEST_CONFIG,
        "test_access_policy": "No test arrays were read until both configurations and the 0.55 validation-selected scale were locked.",
    }
    (RESULTS / "selected_models.json").write_text(json.dumps(selection, indent=2) + "\n")
    (RESULTS / "final_metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    np.savez_compressed(
        RESULTS / "test_predictions.npz",
        true_flags=CLASS_VALUES[y_test],
        small_predicted_flags=CLASS_VALUES[small_test_probability.argmax(axis=1)],
        best_predicted_flags=CLASS_VALUES[best_test_probability.argmax(axis=1)],
        small_probabilities=small_test_probability,
        best_probabilities=best_test_probability,
        shots=data["test_shots"], frames=data["test_frames"], points=data["test_points"],
    )

    small_eval_bundle = {
        **small, "feature_names": data["compact_names"], "class_values": CLASS_VALUES,
        "config": SMALL_CONFIG, "training_scope": "training split only (evaluation artifact)",
    }
    joblib.dump(small_eval_bundle, MODELS / "small_decision_tree_evaluation.joblib", compress=3)
    checkpoint.update(
        {
            "failure_probability_scale": BEST_FAILURE_SCALE,
            "class_values": CLASS_VALUES,
            "training_scope": "training split only; epoch selected on validation (evaluation artifact)",
        }
    )
    torch.save(checkpoint, MODELS / "best_profile_mlp_evaluation.pt")

    # Refit deployable artifacts on all development data while preserving test isolation.
    x_development_compact = np.concatenate([data["train_compact"], data["validation_compact"]])
    x_development_profile = np.concatenate([data["train_profile"], data["validation_profile"]])
    y_development = np.concatenate([y_train, y_validation])
    small_full = fit_small(x_development_compact, y_development)
    joblib.dump(
        {
            **small_full, "feature_names": data["compact_names"], "class_values": CLASS_VALUES,
            "config": SMALL_CONFIG,
            "training_scope": "mission-2512 train+validation; held-out test excluded",
        },
        MODELS / "small_decision_tree.joblib", compress=3,
    )
    full_checkpoint = fit_full_development_neural(x_development_profile, y_development)
    full_checkpoint["feature_names"] = data["profile_names"]
    full_checkpoint["class_values"] = CLASS_VALUES
    torch.save(full_checkpoint, MODELS / "best_profile_mlp.pt")

    model_sizes = {
        path.name: path.stat().st_size
        for path in MODELS.iterdir() if path.is_file()
    }
    all_metrics["artifact_bytes"] = model_sizes
    (RESULTS / "final_metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")

    with (RESULTS / "test_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "exact_accuracy", "macro_f1_observed", "weighted_f1", "failure_precision", "failure_recall", "failure_f1", "failure_pr_auc", "failure_roc_auc"]
        )
        for key in ("baseline", "small", "best"):
            exact = all_metrics[key]["test"]["exact"]
            binary = all_metrics[key]["test"]["binary_failure"]
            writer.writerow(
                [key, exact["accuracy"], exact["macro_f1_observed"], exact["weighted_f1"],
                 binary["precision"], binary["recall"], binary["f1"], binary["pr_auc"], binary["roc_auc"]]
            )
    with (RESULTS / "test_per_class.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "flag", "meaning", "support", "precision", "recall", "f1"])
        for key in ("small", "best"):
            for flag in CLASS_VALUES:
                values = all_metrics[key]["test"]["per_class"][str(int(flag))]
                writer.writerow([key, int(flag), FLAG_NAMES[int(flag)], values["support"], values["precision"], values["recall"], values["f1"]])

    search_records = [
        json.loads(line) for line in (HERE / "validation_search.jsonl").read_text().splitlines()
        if line.strip()
    ]
    with (RESULTS / "validation_search_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "status", "family", "feature_set", "macro_f1", "accuracy", "weighted_f1", "failure_pr_auc", "fit_seconds"])
        for record in search_records:
            config = record["config"]
            if record["status"] == "ok":
                exact = record["validation"]["exact"]
                binary = record["validation"]["binary_failure"]
                writer.writerow([record["id"], "ok", config["family"], config["variant"], exact["macro_f1_observed"], exact["accuracy"], exact["weighted_f1"], binary["pr_auc"], record["fit_seconds"]])
            else:
                writer.writerow([record["id"], "error", config["family"], config["variant"], "", "", "", "", record["fit_seconds"]])

    save_figures(all_metrics, y_test, probabilities)
    print(json.dumps({
        key: all_metrics[key]["test"] for key in ("baseline", "small", "best")
    }, indent=2))


if __name__ == "__main__":
    main()

