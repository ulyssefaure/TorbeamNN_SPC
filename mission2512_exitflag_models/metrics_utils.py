"""Shared metrics and label definitions for mission-2512 exit-flag models."""

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


CLASS_VALUES = np.array([0, 1, 2, 3, 4, 8, 10, 100], dtype=np.int64)
FLAG_NAMES = {
    0: "normal",
    1: "no plasma intersection",
    2: "crossed, no absorption",
    3: "plasma cutoff",
    4: "integrator failure",
    8: "boundary cutoff",
    10: "invalid absorption point",
    100: "timeout / no output file",
}


def encode_flags(flags: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(CLASS_VALUES)}
    try:
        return np.asarray([lookup[int(value)] for value in flags], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unexpected exit flag: {exc.args[0]}") from exc


def decode_flags(encoded: np.ndarray) -> np.ndarray:
    return CLASS_VALUES[np.asarray(encoded, dtype=np.int64)]


def class_sample_weights(encoded: np.ndarray, power: float, cap: float = 30.0) -> np.ndarray:
    """Return tempered inverse-frequency weights with mean one.

    ``power=0`` gives uniform weights and ``power=1`` gives conventional
    inverse-frequency balancing.  Capping prevents the sole flag-4 example
    from dominating a fit.
    """
    encoded = np.asarray(encoded, dtype=np.int64)
    counts = np.bincount(encoded, minlength=len(CLASS_VALUES)).astype(float)
    weights = np.ones_like(counts)
    present = counts > 0
    weights[present] = (len(encoded) / (present.sum() * counts[present])) ** power
    weights = np.minimum(weights, cap)
    sample = weights[encoded]
    return sample / sample.mean()


def evaluate_predictions(
    y_true_encoded: np.ndarray,
    y_pred_encoded: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    """Compute exact-code and success/failure metrics.

    Macro F1 over observed labels is the model-selection metric.  The second
    all-eight-class value makes the data-coverage limitation explicit.
    """
    y_true_encoded = np.asarray(y_true_encoded, dtype=np.int64)
    y_pred_encoded = np.asarray(y_pred_encoded, dtype=np.int64)
    observed = np.unique(y_true_encoded)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_encoded,
        y_pred_encoded,
        labels=np.arange(len(CLASS_VALUES)),
        zero_division=0,
    )
    true_failure = y_true_encoded != 0
    pred_failure = y_pred_encoded != 0
    result = {
        "n_samples": int(len(y_true_encoded)),
        "true_flag_counts": {
            str(int(flag)): int(count)
            for flag, count in sorted(Counter(decode_flags(y_true_encoded)).items())
        },
        "predicted_flag_counts": {
            str(int(flag)): int(count)
            for flag, count in sorted(Counter(decode_flags(y_pred_encoded)).items())
        },
        "exact": {
            "accuracy": float(accuracy_score(y_true_encoded, y_pred_encoded)),
            "balanced_accuracy_observed": float(
                balanced_accuracy_score(y_true_encoded, y_pred_encoded)
            ),
            "macro_f1_observed": float(
                f1_score(y_true_encoded, y_pred_encoded, labels=observed, average="macro", zero_division=0)
            ),
            "macro_f1_all_8": float(
                f1_score(
                    y_true_encoded,
                    y_pred_encoded,
                    labels=np.arange(len(CLASS_VALUES)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(f1_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true_encoded, y_pred_encoded)),
        },
        "binary_failure": {
            "prevalence": float(np.mean(true_failure)),
            "accuracy": float(accuracy_score(true_failure, pred_failure)),
            "balanced_accuracy": float(balanced_accuracy_score(true_failure, pred_failure)),
            "precision": float(precision_score(true_failure, pred_failure, zero_division=0)),
            "recall": float(recall_score(true_failure, pred_failure, zero_division=0)),
            "f1": float(f1_score(true_failure, pred_failure, zero_division=0)),
            "mcc": float(matthews_corrcoef(true_failure, pred_failure)),
        },
        "per_class": {
            str(int(flag)): {
                "meaning": FLAG_NAMES[int(flag)],
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, flag in enumerate(CLASS_VALUES)
        },
        "confusion_matrix": confusion_matrix(
            y_true_encoded, y_pred_encoded, labels=np.arange(len(CLASS_VALUES))
        ).tolist(),
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(y_true_encoded), len(CLASS_VALUES)):
            raise ValueError(f"Unexpected probability shape {probabilities.shape}")
        p_failure = 1.0 - probabilities[:, 0]
        result["binary_failure"].update(
            {
                "pr_auc": float(average_precision_score(true_failure, p_failure)),
                "roc_auc": float(roc_auc_score(true_failure, p_failure)),
                "brier": float(brier_score_loss(true_failure, p_failure)),
            }
        )
    return result

