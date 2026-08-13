"""Shared grouped-label definitions and metrics for mission 2512."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


GROUP_IDS = np.arange(5, dtype=np.int64)
GROUP_NAMES = np.array(
    [
        "normal",
        "no intersection",
        "crossed, no absorption",
        "cutoff",
        "failed run",
    ]
)
GROUP_SHORT_NAMES = np.array(
    ["Normal", "No intersection", "No absorption", "Cutoff", "Failed run"]
)
GROUP_SOURCE_FLAGS = {
    0: [0],
    1: [1],
    2: [2],
    3: [3, 8],
    4: [4, 7, 10, 100],
}
RAW_TO_GROUP = {
    raw_flag: group_id
    for group_id, raw_flags in GROUP_SOURCE_FLAGS.items()
    for raw_flag in raw_flags
}


def group_raw_flags(raw_flags: np.ndarray) -> np.ndarray:
    """Map raw/wrapper TORBEAM flags to the requested five groups."""
    try:
        return np.asarray([RAW_TO_GROUP[int(value)] for value in raw_flags], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"No grouped-label mapping for raw exit flag {exc.args[0]}") from exc


def class_sample_weights(
    labels: np.ndarray,
    power: float,
    cap: float = 30.0,
) -> np.ndarray:
    """Tempered inverse-frequency sample weights, normalized to mean one."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(GROUP_IDS)).astype(float)
    present = counts > 0
    class_weights = np.ones(len(GROUP_IDS), dtype=float)
    class_weights[present] = (
        len(labels) / (present.sum() * counts[present])
    ) ** power
    class_weights = np.minimum(class_weights, cap)
    sample_weights = class_weights[labels]
    return sample_weights / sample_weights.mean()


def apply_non_normal_scale(probabilities: np.ndarray, scale: float) -> np.ndarray:
    """Apply one validation-selected prior correction to all non-normal groups."""
    adjusted = np.asarray(probabilities, dtype=float).copy()
    adjusted[:, 1:] *= float(scale)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def evaluate_grouped(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    """Evaluate five-class prediction and normal versus any-non-normal use."""
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    support_labels = np.unique(true_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=GROUP_IDS,
        zero_division=0,
    )
    true_non_normal = true_labels != 0
    predicted_non_normal = predicted_labels != 0
    result = {
        "n_samples": int(len(true_labels)),
        "true_group_counts": {
            GROUP_NAMES[int(label)]: int(count)
            for label, count in sorted(Counter(true_labels).items())
        },
        "predicted_group_counts": {
            GROUP_NAMES[int(label)]: int(count)
            for label, count in sorted(Counter(predicted_labels).items())
        },
        "grouped": {
            "accuracy": float(accuracy_score(true_labels, predicted_labels)),
            "balanced_accuracy_observed": float(np.mean(recall[support > 0])),
            "macro_f1_observed": float(
                f1_score(
                    true_labels,
                    predicted_labels,
                    labels=support_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1_all_5": float(
                f1_score(
                    true_labels,
                    predicted_labels,
                    labels=GROUP_IDS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
            ),
            "mcc": float(matthews_corrcoef(true_labels, predicted_labels)),
            "observed_group_count": int(len(support_labels)),
        },
        "non_normal": {
            "prevalence": float(np.mean(true_non_normal)),
            "accuracy": float(accuracy_score(true_non_normal, predicted_non_normal)),
            "balanced_accuracy": float(
                balanced_accuracy_score(true_non_normal, predicted_non_normal)
            ),
            "precision": float(
                precision_score(true_non_normal, predicted_non_normal, zero_division=0)
            ),
            "recall": float(
                recall_score(true_non_normal, predicted_non_normal, zero_division=0)
            ),
            "f1": float(f1_score(true_non_normal, predicted_non_normal, zero_division=0)),
            "mcc": float(matthews_corrcoef(true_non_normal, predicted_non_normal)),
        },
        "per_group": {
            GROUP_NAMES[index]: {
                "group_id": int(index),
                "source_flags": GROUP_SOURCE_FLAGS[index],
                "support": int(support[index]),
                "precision": float(precision[index]) if support[index] else None,
                "recall": float(recall[index]) if support[index] else None,
                "f1": float(f1[index]) if support[index] else None,
            }
            for index in GROUP_IDS
        },
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=GROUP_IDS
        ).tolist(),
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(true_labels), len(GROUP_IDS)):
            raise ValueError(f"Unexpected grouped probability shape {probabilities.shape}")
        p_non_normal = 1.0 - probabilities[:, 0]
        result["non_normal"].update(
            {
                "pr_auc": float(
                    average_precision_score(true_non_normal, p_non_normal)
                ),
                "roc_auc": float(roc_auc_score(true_non_normal, p_non_normal)),
                "brier": float(brier_score_loss(true_non_normal, p_non_normal)),
            }
        )
        for group_id, name in enumerate(GROUP_NAMES):
            binary_truth = true_labels == group_id
            if binary_truth.any() and (~binary_truth).any():
                result["per_group"][name]["pr_auc"] = float(
                    average_precision_score(binary_truth, probabilities[:, group_id])
                )
                result["per_group"][name]["roc_auc"] = float(
                    roc_auc_score(binary_truth, probabilities[:, group_id])
                )
            else:
                result["per_group"][name]["pr_auc"] = None
                result["per_group"][name]["roc_auc"] = None
    return result


def best_non_normal_scale(
    probabilities: np.ndarray,
    labels: np.ndarray,
    scales: np.ndarray | None = None,
) -> tuple[float, dict, np.ndarray]:
    """Choose one scalar on validation by macro F1, then accuracy."""
    if scales is None:
        scales = np.arange(0.25, 1.5001, 0.025)
    choices = []
    for scale in scales:
        adjusted = apply_non_normal_scale(probabilities, float(scale))
        metrics = evaluate_grouped(labels, adjusted.argmax(axis=1), adjusted)
        choices.append(
            (
                metrics["grouped"]["macro_f1_observed"],
                metrics["grouped"]["accuracy"],
                -abs(float(scale) - 1.0),
                float(scale),
                metrics,
                adjusted,
            )
        )
    _, _, _, scale, metrics, adjusted = max(choices, key=lambda item: item[:3])
    return scale, metrics, adjusted
