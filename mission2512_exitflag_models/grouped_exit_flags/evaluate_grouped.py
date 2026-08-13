#!/usr/bin/env python3
"""Evaluate locked grouped-label models and create adapted graphs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from grouped_utils import (
    GROUP_IDS,
    GROUP_NAMES,
    GROUP_SHORT_NAMES,
    GROUP_SOURCE_FLAGS,
    apply_non_normal_scale,
    class_sample_weights,
    evaluate_grouped,
    group_raw_flags,
)
from train_validate_grouped import GroupedMLP, aligned_tree_probabilities


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
MODELS = HERE / "models"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def transform_profile(values: np.ndarray, checkpoint: dict) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    statistics = np.asarray(checkpoint["imputer_statistics"], dtype=float)
    transformed = np.where(missing, statistics[None, :], values)
    indicator_features = np.asarray(checkpoint["indicator_features"], dtype=int)
    if len(indicator_features):
        transformed = np.column_stack(
            [transformed, missing[:, indicator_features].astype(float)]
        )
    return (
        (transformed - np.asarray(checkpoint["scaler_mean"]))
        / np.asarray(checkpoint["scaler_scale"])
    ).astype(np.float32)


def mlp_probabilities(values: np.ndarray, checkpoint: dict) -> np.ndarray:
    transformed = transform_profile(values, checkpoint)
    model = GroupedMLP(transformed.shape[1], dropout=checkpoint["config"]["dropout"])
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
    return np.concatenate(chunks)


def failed_run_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth = labels == 4
    prediction = probabilities[:, 4]
    return {
        "prevalence": float(truth.mean()),
        "pr_auc": float(average_precision_score(truth, prediction)),
        "roc_auc": float(roc_auc_score(truth, prediction)),
    }


def cluster_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    repetitions: int = 1000,
) -> dict:
    rng = np.random.default_rng(20260812)
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    values = {
        key: []
        for key in (
            "accuracy", "macro_f1_observed", "weighted_f1",
            "non_normal_pr_auc", "failed_run_pr_auc",
        )
    }
    for _ in range(repetitions):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in selected_groups])
        metrics = evaluate_grouped(
            labels[selected], probabilities[selected].argmax(axis=1), probabilities[selected]
        )
        failed = failed_run_metrics(labels[selected], probabilities[selected])
        values["accuracy"].append(metrics["grouped"]["accuracy"])
        values["macro_f1_observed"].append(metrics["grouped"]["macro_f1_observed"])
        values["weighted_f1"].append(metrics["grouped"]["weighted_f1"])
        values["non_normal_pr_auc"].append(metrics["non_normal"]["pr_auc"])
        values["failed_run_pr_auc"].append(failed["pr_auc"])
    return {
        name: {
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def refit_tree_deployment(
    data: np.lib.npyio.NpzFile,
    labels: np.ndarray,
    evaluation_bundle: dict,
) -> None:
    values = np.concatenate([data["train_compact"], data["validation_compact"]])
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    transformed = imputer.fit_transform(values)
    config = evaluation_bundle["config"]
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        criterion=config["criterion"],
        random_state=42,
    )
    model.fit(
        transformed,
        labels,
        sample_weight=class_sample_weights(labels, config["class_weight_power"]),
    )
    bundle = {
        **evaluation_bundle,
        "model": model,
        "imputer": imputer,
        "training_scope": "mission-2512 train+validation; held-out test excluded",
    }
    joblib.dump(bundle, MODELS / "small_grouped_tree.joblib", compress=3)


def refit_mlp_deployment(
    data: np.lib.npyio.NpzFile,
    labels: np.ndarray,
    evaluation_checkpoint: dict,
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    values = np.concatenate([data["train_profile"], data["validation_profile"]])
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(values)).astype(np.float32)
    config = evaluation_checkpoint["config"]
    model = GroupedMLP(transformed.shape[1], dropout=config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-4)
    sample_weights = class_sample_weights(labels, config["class_weight_power"])
    class_weights = torch.from_numpy(
        np.asarray([sample_weights[labels == label][0] for label in GROUP_IDS], dtype=np.float32)
    )
    dataset = TensorDataset(torch.from_numpy(transformed), torch.from_numpy(labels))
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    for _ in range(config["epochs"]):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features), targets, weight=class_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    checkpoint = {
        **evaluation_checkpoint,
        "state_dict": model.state_dict(),
        "imputer_statistics": imputer.statistics_,
        "indicator_features": imputer.indicator_.features_,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "training_scope": "mission-2512 train+validation; held-out test excluded",
    }
    torch.save(checkpoint, MODELS / "best_grouped_profile_mlp.pt")


def plot_confusion_pair(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), constrained_layout=True)
    image = None
    for axis, key, title in zip(
        axes,
        ("small_tree", "best_profile_mlp"),
        ("Small depth-8 tree", "Best profile MLP"),
    ):
        matrix = np.asarray(metrics[key]["test"]["confusion_matrix"], dtype=float)
        support = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, support, out=np.zeros_like(matrix), where=support > 0)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in GROUP_IDS:
            for column in GROUP_IDS:
                count = int(matrix[row, column])
                if count:
                    axis.text(
                        column, row, str(count), ha="center", va="center", fontsize=9,
                        color="white" if normalized[row, column] > 0.55 else "black",
                    )
        axis.set_xticks(GROUP_IDS, GROUP_SHORT_NAMES, rotation=27, ha="right")
        axis.set_yticks(GROUP_IDS, GROUP_SHORT_NAMES)
        axis.set_xlabel("Predicted group")
        axis.set_ylabel("True group")
        axis.set_title(title)
    fig.colorbar(image, ax=axes, shrink=0.84, label="Fraction within true group")
    fig.suptitle("Mission 2512 grouped exit flags — held-out frames", fontsize=15)
    fig.savefig(FIGURES / "grouped_confusion_matrices.png", dpi=220)
    fig.savefig(FIGURES / "grouped_confusion_matrices.pdf")
    plt.close(fig)


def plot_distribution(counts: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 4.9))
    bars = axis.bar(GROUP_SHORT_NAMES, counts, color="#4472C4")
    axis.set_yscale("log")
    axis.set_ylabel("Count (log scale)")
    axis.set_title("Mission 2512 grouped exit-label distribution")
    axis.bar_label(bars, labels=[f"{int(value):,}" for value in counts], padding=3)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "grouped_class_distribution.png", dpi=220)
    fig.savefig(FIGURES / "grouped_class_distribution.pdf")
    plt.close(fig)


def plot_per_group_f1(metrics: dict) -> None:
    supports = [
        metrics["best_profile_mlp"]["test"]["per_group"][name]["support"]
        for name in GROUP_NAMES
    ]
    x = np.arange(len(GROUP_IDS)); width = 0.36
    fig, axis = plt.subplots(figsize=(10.0, 5.4))
    for offset, (key, label, color) in enumerate(
        (
            ("small_tree", "Small tree", "#4472C4"),
            ("best_profile_mlp", "Best MLP", "#ED7D31"),
        )
    ):
        values = []
        for name in GROUP_NAMES:
            value = metrics[key]["test"]["per_group"][name]["f1"]
            values.append(np.nan if value is None else value)
        bars = axis.bar(x + (offset - 0.5) * width, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.text(
                    bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )
    # Explicitly mark unsupported test groups rather than graphing zero.
    for index, support in enumerate(supports):
        if support == 0:
            axis.text(index, 0.08, "N/A\n(no test cases)", ha="center", va="center", fontsize=9)
    axis.set_xticks(x, [f"{name}\n(n={support})" for name, support in zip(GROUP_SHORT_NAMES, supports)])
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("F1 score")
    axis.set_title("Grouped-label F1 on held-out frames")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "grouped_per_class_f1.png", dpi=220)
    fig.savefig(FIGURES / "grouped_per_class_f1.pdf")
    plt.close(fig)


def plot_score_comparison(metrics: dict) -> None:
    keys = ("baseline", "small_tree", "best_profile_mlp")
    labels = ("Majority baseline", "Small tree", "Best MLP")
    specs = (
        ("Accuracy", lambda item: item["test"]["grouped"]["accuracy"]),
        ("Macro F1\n(4 supported groups)", lambda item: item["test"]["grouped"]["macro_f1_observed"]),
        ("Weighted F1", lambda item: item["test"]["grouped"]["weighted_f1"]),
        ("Failed-run PR-AUC", lambda item: item["test"]["failed_run"]["pr_auc"]),
    )
    x = np.arange(len(specs)); width = 0.24
    fig, axis = plt.subplots(figsize=(10.0, 5.5))
    for offset, (key, label) in enumerate(zip(keys, labels)):
        values = [function(metrics[key]) for _, function in specs]
        bars = axis.bar(x + (offset - 1) * width, values, width, label=label)
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axis.set_xticks(x, [name for name, _ in specs])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Grouped exit-flag models — held-out comparison")
    axis.legend(loc="upper center", ncol=3)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "grouped_model_comparison.png", dpi=220)
    fig.savefig(FIGURES / "grouped_model_comparison.pdf")
    plt.close(fig)


def plot_failed_run_pr(labels: np.ndarray, probabilities: dict, metrics: dict) -> None:
    truth = labels == 4
    fig, axis = plt.subplots(figsize=(7.0, 5.5))
    for key, label, color in (
        ("small_tree", "Small tree", "#4472C4"),
        ("best_profile_mlp", "Best MLP", "#ED7D31"),
    ):
        precision, recall, _ = precision_recall_curve(truth, probabilities[key][:, 4])
        ap = metrics[key]["test"]["failed_run"]["pr_auc"]
        axis.plot(recall, precision, label=f"{label} (AP={ap:.3f})", color=color)
    prevalence = truth.mean()
    axis.axhline(
        prevalence, color="0.4", linestyle="--",
        label=f"Prevalence ({prevalence:.3f})",
    )
    axis.set(xlabel="Failed-run recall", ylabel="Failed-run precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.set_title("Failed-run detection: flags 4, 7, 10, 100")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "grouped_failed_run_precision_recall.png", dpi=220)
    fig.savefig(FIGURES / "grouped_failed_run_precision_recall.pdf")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = np.load(PARENT / "dataset_cache.npz")
    selection = json.loads((RESULTS / "selection.json").read_text())
    # The configurations above were serialized before these test arrays are accessed.
    y_test = group_raw_flags(data["test_labels"])
    y_development = group_raw_flags(
        np.concatenate([data["train_labels"], data["validation_labels"]])
    )
    tree_bundle = joblib.load(MODELS / "small_grouped_tree_evaluation.joblib")
    x_test_compact = tree_bundle["imputer"].transform(data["test_compact"])
    tree_probability = aligned_tree_probabilities(tree_bundle["model"], x_test_compact)
    tree_probability = apply_non_normal_scale(
        tree_probability, tree_bundle["non_normal_probability_scale"]
    )
    mlp_checkpoint = torch.load(
        MODELS / "best_grouped_profile_mlp_evaluation.pt", map_location="cpu"
    )
    mlp_probability = mlp_probabilities(data["test_profile"], mlp_checkpoint)
    mlp_probability = apply_non_normal_scale(
        mlp_probability, mlp_checkpoint["non_normal_probability_scale"]
    )
    baseline_probability = np.zeros((len(y_test), len(GROUP_IDS)), dtype=float)
    baseline_probability[:, 0] = 1.0
    probabilities = {
        "baseline": baseline_probability,
        "small_tree": tree_probability,
        "best_profile_mlp": mlp_probability,
    }
    metrics = {}
    for key, probability in probabilities.items():
        metrics[key] = {
            "test": evaluate_grouped(
                y_test, probability.argmax(axis=1), probability
            )
        }
        metrics[key]["test"]["failed_run"] = failed_run_metrics(
            y_test, probability
        )
    metrics["small_tree"]["validation"] = selection["small_tree"]["validation"]
    metrics["best_profile_mlp"]["validation"] = selection["best_profile_mlp"]["validation"]
    groups = np.asarray(
        [f"{shot}:{frame}" for shot, frame in zip(data["test_shots"], data["test_frames"])]
    )
    for key in ("small_tree", "best_profile_mlp"):
        metrics[key]["cluster_bootstrap_95_ci"] = cluster_bootstrap(
            y_test, probabilities[key], groups
        )
        per_shot = {}
        for shot in np.unique(data["test_shots"]):
            mask = data["test_shots"] == shot
            shot_metrics = evaluate_grouped(
                y_test[mask],
                probabilities[key][mask].argmax(axis=1),
                probabilities[key][mask],
            )
            shot_metrics["failed_run"] = failed_run_metrics(
                y_test[mask], probabilities[key][mask]
            )
            per_shot[str(int(shot))] = shot_metrics
        metrics[key]["per_shot_test"] = per_shot

    np.savez_compressed(
        RESULTS / "test_predictions.npz",
        true_group=y_test,
        small_predicted_group=tree_probability.argmax(axis=1),
        best_predicted_group=mlp_probability.argmax(axis=1),
        small_probabilities=tree_probability,
        best_probabilities=mlp_probability,
        raw_exit_flags=data["test_labels"],
        shots=data["test_shots"],
        frames=data["test_frames"],
        points=data["test_points"],
    )
    with (RESULTS / "test_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model", "accuracy", "macro_f1_observed", "macro_f1_all_5",
                "weighted_f1", "non_normal_precision", "non_normal_recall",
                "non_normal_pr_auc", "failed_run_pr_auc",
            ]
        )
        for key in ("baseline", "small_tree", "best_profile_mlp"):
            grouped = metrics[key]["test"]["grouped"]
            non_normal = metrics[key]["test"]["non_normal"]
            failed = metrics[key]["test"]["failed_run"]
            writer.writerow(
                [
                    key, grouped["accuracy"], grouped["macro_f1_observed"],
                    grouped["macro_f1_all_5"], grouped["weighted_f1"],
                    non_normal["precision"], non_normal["recall"],
                    non_normal["pr_auc"], failed["pr_auc"],
                ]
            )
    with (RESULTS / "test_per_group.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "group", "source_flags", "support", "precision", "recall", "f1", "pr_auc"]
        )
        for key in ("small_tree", "best_profile_mlp"):
            for name in GROUP_NAMES:
                item = metrics[key]["test"]["per_group"][name]
                writer.writerow(
                    [key, name, "+".join(map(str, item["source_flags"])), item["support"],
                     item["precision"], item["recall"], item["f1"], item["pr_auc"]]
                )

    # Refit deployable copies on all development rows, never on held-out test.
    refit_tree_deployment(data, y_development, tree_bundle)
    refit_mlp_deployment(data, y_development, mlp_checkpoint)
    metrics["artifact_bytes"] = {
        path.name: path.stat().st_size for path in MODELS.iterdir() if path.is_file()
    }
    (RESULTS / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    full_counts = np.bincount(
        group_raw_flags(
            np.concatenate([data["train_labels"], data["validation_labels"], data["test_labels"]])
        ),
        minlength=len(GROUP_IDS),
    )
    plot_confusion_pair(metrics)
    plot_distribution(full_counts)
    plot_per_group_f1(metrics)
    plot_score_comparison(metrics)
    plot_failed_run_pr(y_test, probabilities, metrics)
    print(json.dumps({key: metrics[key]["test"] for key in metrics if key != "artifact_bytes"}, indent=2))


if __name__ == "__main__":
    main()

