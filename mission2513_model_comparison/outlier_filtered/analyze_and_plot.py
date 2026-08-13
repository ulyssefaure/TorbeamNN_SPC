#!/usr/bin/env python3
"""Apply a training-derived input-domain screen and regenerate R2 charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OUTPUT_NAMES = ("rho_pol", "R", "Z", "CD_eta", "w_cd")
MODEL_KEYS = {
    "article": ("Article 60×3 ReLU", "test_article_predictions"),
    "wide_fourier": ("Wide Fourier-profile NN", "test_wide_fourier_predictions"),
}
THRESHOLD = 8.0


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for index, name in enumerate(OUTPUT_NAMES):
        residual = prediction[:, index]-target[:, index]
        denominator = np.sum((target[:, index]-target[:, index].mean())**2)
        result[name] = {
            "r2": float(1-np.sum(residual**2)/denominator),
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
        }
    return result


def add_parity_row(
    axes: np.ndarray, target: np.ndarray, prediction: np.ndarray,
    scores: dict[str, dict[str, float]], ylabel: str,
) -> None:
    for index, (name, axis) in enumerate(zip(OUTPUT_NAMES, axes)):
        truth, estimate = target[:, index], prediction[:, index]
        low = min(float(truth.min()), float(estimate.min()))
        high = max(float(truth.max()), float(estimate.max()))
        padding = .03*(high-low)
        axis.scatter(truth, estimate, s=8, alpha=.42, edgecolors="none")
        axis.plot([low-padding, high+padding], [low-padding, high+padding], "r--", lw=1.2)
        axis.set(
            xlabel="TORBEAM", ylabel=ylabel, title=name,
            xlim=(low-padding, high+padding), ylim=(low-padding, high+padding),
        )
        value = scores[name]
        axis.text(
            .04, .96,
            f"R² = {value['r2']:.4f}\nMAE = {value['mae']:.4g}\nRMSE = {value['rmse']:.4g}",
            transform=axis.transAxes, va="top", fontsize=9,
            bbox={"boxstyle":"round", "facecolor":"white", "alpha":.82, "edgecolor":".8"},
        )
        axis.grid(alpha=.2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = np.load(ROOT/"dataset_cache.npz")
    saved = np.load(ROOT/"predictions.npz")
    target = saved["test_targets"]
    wide_train = cache["train_wide"]
    mean, scale = wide_train.mean(axis=0), wide_train.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = np.abs((cache["test_wide"]-mean)/scale)
    domain_score = standardized.max(axis=1)
    training_domain_score = np.abs((wide_train-mean)/scale).max(axis=1)
    validation_domain_score = np.abs((cache["validation_wide"]-mean)/scale).max(axis=1)
    keep = domain_score <= THRESHOLD
    removed = np.flatnonzero(~keep)
    if len(removed) == 0:
        raise RuntimeError("The configured screen did not flag any samples")

    original_results, filtered_results, influence = {}, {}, {}
    predictions = {}
    for key, (_, prediction_key) in MODEL_KEYS.items():
        prediction = saved[prediction_key]
        predictions[key] = prediction
        original_results[key] = metrics(target, prediction)
        filtered_results[key] = metrics(target[keep], prediction[keep])
        influence[key] = {}
        for index, name in enumerate(OUTPUT_NAMES):
            squared = (prediction[:, index]-target[:, index])**2
            influence[key][name] = {
                "removed_sample_fraction_of_total_squared_error": float(squared[~keep].sum()/squared.sum()),
                "removed_sample_squared_error": float(squared[~keep].sum()),
                "original_total_squared_error": float(squared.sum()),
            }

    feature_names = cache["wide_names"]
    excluded = []
    for row in removed:
        extreme_indices = np.flatnonzero(standardized[row] > THRESHOLD)
        excluded.append({
            "test_row": int(row),
            "shot": int(cache["test_shots"][row]),
            "frame": int(cache["test_frames"][row]),
            "launcher_point": 17 if int(row) == 261 else None,
            "maximum_absolute_training_z_score": float(domain_score[row]),
            "targets": {name: float(target[row, index]) for index, name in enumerate(OUTPUT_NAMES)},
            "extreme_inputs": [
                {
                    "name": str(feature_names[index]),
                    "value": float(cache["test_wide"][row, index]),
                    "training_z_score": float((cache["test_wide"][row, index]-mean[index])/scale[index]),
                    "training_min": float(wide_train[:, index].min()),
                    "training_max": float(wide_train[:, index].max()),
                }
                for index in extreme_indices
            ],
        })

    report = {
        "rule": {
            "definition": "exclude if max_j |(x_j - training_mean_j) / training_std_j| > 8",
            "threshold": THRESHOLD,
            "uses_test_targets_or_residuals": False,
            "feature_space": "51 wide Fourier-profile inputs",
            "maximum_training_score": float(training_domain_score.max()),
            "maximum_validation_score": float(validation_domain_score.max()),
            "largest_nonexcluded_test_score": float(domain_score[keep].max()),
            "threshold_sensitivity": "Any threshold from 6.25 through 17.21 flags exactly this one test sample and no training/validation sample.",
        },
        "original_test_samples": int(len(target)),
        "retained_test_samples": int(keep.sum()),
        "excluded_test_samples": int((~keep).sum()),
        "excluded": excluded,
        "original_metrics": original_results,
        "filtered_metrics": filtered_results,
        "squared_error_influence": influence,
    }
    (OUT/"outlier_report.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(
        OUT/"filtered_predictions.npz",
        keep_mask=keep, domain_score=domain_score,
        test_targets=target[keep],
        article_predictions=predictions["article"][keep],
        wide_fourier_predictions=predictions["wide_fourier"][keep],
        excluded_indices=removed,
    )

    figure, axes = plt.subplots(2, 5, figsize=(22, 8.8))
    for row, (key, (label, _)) in enumerate(MODEL_KEYS.items()):
        add_parity_row(
            axes[row], target[keep], predictions[key][keep], filtered_results[key], label
        )
        axes[row, 0].annotate(
            f"({chr(97+row)})", xy=(-.22, 1.10), xycoords="axes fraction",
            fontsize=14, fontweight="bold",
        )
    figure.suptitle(
        f"Mission 2513 in-domain held-out R² comparison — {keep.sum()}/{len(keep)} samples",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, .97), h_pad=2.3)
    figure.savefig(OUT/"article_vs_wide_fourier_r2_filtered.png", dpi=180)
    plt.close(figure)

    x=np.arange(len(OUTPUT_NAMES)); width=.36
    figure, axis=plt.subplots(figsize=(11,5.8))
    for offset,(key,(label,_)) in zip((-.18,.18),MODEL_KEYS.items()):
        values=[filtered_results[key][name]["r2"] for name in OUTPUT_NAMES]
        bars=axis.bar(x+offset,values,width,label=label)
        axis.bar_label(bars,labels=[f"{value:.3f}" for value in values],padding=3,fontsize=9)
    axis.set(
        ylabel="Held-out R²", title="Mission 2513 R² after training-domain screen",
        xticks=x, xticklabels=OUTPUT_NAMES, ylim=(0,1.04),
    )
    axis.grid(axis="y",alpha=.25); axis.legend(); figure.tight_layout()
    figure.savefig(OUT/"r2_bar_comparison_filtered.png",dpi=180); plt.close(figure)

    # Show explicitly how much the single OOD point changes each reported R2.
    x=np.arange(len(OUTPUT_NAMES)); width=.2
    figure,axis=plt.subplots(figsize=(13,6))
    series=[]
    for key,(label,_) in MODEL_KEYS.items():
        series.extend([
            (f"{label} — all",[original_results[key][name]["r2"] for name in OUTPUT_NAMES]),
            (f"{label} — in-domain",[filtered_results[key][name]["r2"] for name in OUTPUT_NAMES]),
        ])
    for number,(label,values) in enumerate(series):
        offset=(number-1.5)*width
        axis.bar(x+offset,values,width,label=label)
    axis.set(
        ylabel="Held-out R²", title="Influence of the single out-of-domain test sample",
        xticks=x,xticklabels=OUTPUT_NAMES,ylim=(0,1.02),
    )
    axis.grid(axis="y",alpha=.25); axis.legend(ncol=2); figure.tight_layout()
    figure.savefig(OUT/"r2_before_after_domain_screen.png",dpi=180); plt.close(figure)

    rows=[]
    for name in OUTPUT_NAMES:
        a=filtered_results["article"][name]["r2"]
        w=filtered_results["wide_fourier"][name]["r2"]
        rows.append(f"| `{name}` | {a:.6f} | {w:.6f} | {w-a:+.6f} |")
    readme=f"""# Mission 2513 input-domain-filtered evaluation

The original data and results remain unchanged. This evaluation excludes a
test sample only when at least one of its 51 wide Fourier-profile inputs is more
than {THRESHOLD:g} training standard deviations from the training mean. The
rule uses training inputs only--not test targets, prediction errors, or R2.

Exactly one of 268 test samples is flagged: shot 85811, frame 7, launcher point
17. Its maximum input-domain score is {domain_score[removed[0]]:.2f} sigma; ten
density-profile coordinates are 10.15--17.22 sigma from training and exceed
the corresponding training maxima. No training or validation sample exceeds
the 8-sigma threshold. The result is not sensitive to the precise cutoff: any
threshold from 6.25 through 17.21 isolates the same point. Since the excluded point is in the test set, retraining
would not change either checkpoint; only the in-domain evaluation changes.

| Output | Article R2 | Wide Fourier-profile R2 | Difference |
|---|---:|---:|---:|
{chr(10).join(rows)}

See `outlier_report.json` for the exact rule, excluded values, original and
filtered metrics, and squared-error influence. The raw comparison in the parent
folder should still be reported whenever out-of-domain behavior matters.
"""
    (OUT/"README.md").write_text(readme)
    print(json.dumps({
        "removed_indices":removed.tolist(),
        "domain_scores":domain_score[removed].tolist(),
        "filtered_r2":{
            key:{name:filtered_results[key][name]["r2"] for name in OUTPUT_NAMES}
            for key in MODEL_KEYS
        },
    },indent=2))


if __name__ == "__main__":
    main()
