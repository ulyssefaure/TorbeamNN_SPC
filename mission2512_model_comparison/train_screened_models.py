#!/usr/bin/env python3
"""Retrain article and wide Fourier-profile models after the 8-sigma screen."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"mission2513_model_comparison"))
from train_models import (  # noqa: E402
    ArticleNet,
    WideFourierProfileNet,
    add_parity_row,
    save_model,
    train_model,
)


OUTPUT_NAMES = np.array(["rho_pol", "R", "Z", "CD_eta", "w_cd"])
THRESHOLD = 8.0


def wide_features(profile: np.ndarray) -> np.ndarray:
    angles = profile[:, :2]
    harmonics = np.column_stack([
        function(harmonic*angles[:, angle_index])
        for harmonic in range(1, 5)
        for angle_index in range(2)
        for function in (np.sin, np.cos)
    ])
    return np.column_stack([profile, harmonics])


def make_figures(
    targets: np.ndarray,
    results: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, np.ndarray]],
) -> None:
    labels = {"article":"Article 60×3 ReLU", "wide_fourier":"Wide Fourier-profile NN"}
    for key in ("article", "wide_fourier"):
        figure, axes = plt.subplots(1, 5, figsize=(22, 4.4))
        add_parity_row(
            axes, targets, predictions[key]["test"], results[key]["test_metrics"], labels[key]
        )
        figure.suptitle(f"{labels[key]} held-out-frame parity — mission 2512", fontsize=14)
        figure.tight_layout(); figure.savefig(OUT/f"{key}_r2.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(2, 5, figsize=(22, 8.8))
    for row, key in enumerate(("article", "wide_fourier")):
        add_parity_row(
            axes[row], targets, predictions[key]["test"], results[key]["test_metrics"], labels[key]
        )
        axes[row, 0].annotate(
            f"({chr(97+row)})", xy=(-.22, 1.10), xycoords="axes fraction",
            fontsize=14, fontweight="bold",
        )
    figure.suptitle("Mission 2512 held-out R² comparison", fontsize=15)
    figure.tight_layout(rect=(0,0,1,.97), h_pad=2.3)
    figure.savefig(OUT/"article_vs_wide_fourier_r2.png", dpi=180); plt.close(figure)

    x=np.arange(len(OUTPUT_NAMES)); width=.36
    figure,axis=plt.subplots(figsize=(11,5.8))
    for offset,key in zip((-.18,.18),("article","wide_fourier")):
        values=[results[key]["test_metrics"][str(name)]["r2"] for name in OUTPUT_NAMES]
        bars=axis.bar(x+offset,values,width,label=labels[key])
        axis.bar_label(bars,labels=[f"{value:.3f}" for value in values],padding=3,fontsize=9)
    axis.set(
        ylabel="Held-out R²", title="Mission 2512 R² comparison",
        xticks=x, xticklabels=OUTPUT_NAMES, ylim=(0,1.04),
    )
    axis.grid(axis="y",alpha=.25); axis.legend(); figure.tight_layout()
    figure.savefig(OUT/"r2_bar_comparison.png",dpi=180); plt.close(figure)


def write_tables(results: dict[str, dict[str, Any]]) -> None:
    rows=[]
    for name in OUTPUT_NAMES:
        article=results["article"]["test_metrics"][str(name)]
        wide=results["wide_fourier"]["test_metrics"][str(name)]
        rows.append({
            "output":str(name),
            "article_r2":article["r2"], "article_mae":article["mae"],
            "wide_r2":wide["r2"], "wide_mae":wide["mae"],
            "r2_difference_wide_minus_article":wide["r2"]-article["r2"],
            "mae_difference_wide_minus_article":wide["mae"]-article["mae"],
        })
    with (OUT/"metrics_table.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=[
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Output & Article $R^2$ & Article MAE & Wide $R^2$ & Wide MAE \\",
        r"\midrule",
    ]
    for row in rows:
        name=row["output"].replace("_",r"\_")
        lines.append(
            f"{name} & {row['article_r2']:.6f} & {row['article_mae']:.6g} & "
            f"{row['wide_r2']:.6f} & {row['wide_mae']:.6g} " + r"\\"
        )
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (OUT/"metrics_table.tex").write_text("\n".join(lines)+"\n")


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(max(1,min(8,os.cpu_count() or 1)))
    cache=np.load(ROOT/"model_search_mission2512"/"dataset_cache.npz")
    original={
        split:{
            "article":cache[f"{split}_article"],
            "wide":wide_features(cache[f"{split}_profile"]),
            "targets":cache[f"{split}_targets"],
            "shots":cache[f"{split}_shots"],
            "frames":cache[f"{split}_frames"],
            "points":cache[f"{split}_points"],
        }
        for split in ("train","validation","test")
    }
    training_wide=original["train"]["wide"]
    domain_mean,domain_scale=training_wide.mean(0),training_wide.std(0)
    domain_scale[domain_scale==0]=1.0
    scores={
        split:np.max(np.abs((values["wide"]-domain_mean)/domain_scale),axis=1)
        for split,values in original.items()
    }
    masks={split:values<=THRESHOLD for split,values in scores.items()}
    parts={
        split:{key:value[masks[split]] for key,value in values.items()}
        for split,values in original.items()
    }
    print("8-sigma screen:",flush=True)
    for split in ("train","validation","test"):
        print(
            f"  {split}: retained {masks[split].sum()}/{len(masks[split])}; "
            f"removed rows {np.flatnonzero(~masks[split]).tolist()}",flush=True
        )

    features={
        "article":{split:values["article"] for split,values in parts.items()},
        "wide_fourier":{split:values["wide"] for split,values in parts.items()},
    }
    targets={split:values["targets"] for split,values in parts.items()}
    configurations={
        "article":{
            "architecture":"14-60-60-60-5 ReLU", "optimizer":"Adam",
            "learning_rate":6.3e-4,"batch_size":128,"epochs":1000,
            "patience":250,"seed":42,
        },
        "wide_fourier":{
            "architecture":"51-256-192-128 SiLU trunk; five 64-1 SiLU heads",
            "optimizer":"AdamW","learning_rate":3.5e-4,"weight_decay":1e-5,
            "batch_size":256,"epochs":140,"patience":35,"seed":212,
        },
    }
    factories:dict[str,Callable[[],nn.Module]]={
        "article":ArticleNet,"wide_fourier":WideFourierProfileNet,
    }
    models:dict[str,nn.Module]={}; predictions={}; scalers={}; results={}
    for key in ("article","wide_fourier"):
        models[key],predictions[key],scalers[key],results[key]=train_model(
            key,factories[key],features[key],targets,configurations[key]
        )
        print(
            f"FINISHED {key}: validation={results[key]['validation_scaled_mse']:.6f}, "
            f"test={results[key]['test_scaled_mse']:.6f}",flush=True
        )

    profile_names=list(cache["profile_names"])
    fourier_names=[
        f"{function}_{angle}_k{harmonic}"
        for harmonic in range(1,5) for angle in ("pol","tor") for function in ("sin","cos")
    ]
    save_model(OUT/"article_model.npz",models["article"],scalers["article"],cache["article_names"],results["article"])
    save_model(
        OUT/"wide_fourier_profile_model.npz",models["wide_fourier"],scalers["wide_fourier"],
        np.asarray(profile_names+fourier_names),results["wide_fourier"],
    )
    prediction_arrays={}
    for split in ("train","validation","test"):
        prediction_arrays[f"{split}_targets"]=targets[split]
        prediction_arrays[f"{split}_article_predictions"]=predictions["article"][split]
        prediction_arrays[f"{split}_wide_fourier_predictions"]=predictions["wide_fourier"][split]
        prediction_arrays[f"{split}_shots"]=parts[split]["shots"]
        prediction_arrays[f"{split}_frames"]=parts[split]["frames"]
        prediction_arrays[f"{split}_points"]=parts[split]["points"]
    np.savez_compressed(OUT/"predictions.npz",**prediction_arrays)

    removed={}
    for split in ("train","validation","test"):
        removed[split]=[
            {
                "original_row":int(row),"shot":int(original[split]["shots"][row]),
                "frame":int(original[split]["frames"][row]),
                "point":int(original[split]["points"][row]),"domain_score":float(scores[split][row]),
            }
            for row in np.flatnonzero(~masks[split])
        ]
    report={
        "mission":2512,
        "screen":{
            "definition":"remove if max_j |(x_j-training_mean_j)/training_std_j| > 8",
            "threshold":THRESHOLD,"uses_targets_or_prediction_residuals":False,
            "removed":removed,
        },
        "sample_counts_before":{split:len(values["targets"]) for split,values in original.items()},
        "sample_counts_after":{split:len(values["targets"]) for split,values in parts.items()},
        "models":results,
    }
    (OUT/"training_report.json").write_text(json.dumps(report,indent=2)+"\n")
    make_figures(targets["test"],results,predictions)
    write_tables(results)

    table_rows=[]
    for name in OUTPUT_NAMES:
        article=results["article"]["test_metrics"][str(name)]
        wide=results["wide_fourier"]["test_metrics"][str(name)]
        table_rows.append(
            f"| `{name}` | {article['r2']:.6f} | {article['mae']:.6g} | "
            f"{wide['r2']:.6f} | {wide['mae']:.6g} |"
        )
    readme=f"""# Mission 2512 article and wide-network comparison

Both models were retrained using the same frame split after applying the
training-input-only 8-sigma screen documented in `training_report.json`. The
screen removes 6 of 20,801 training samples, no validation samples, and 1 of
2,672 test samples. Chart titles intentionally contain no filtering annotation.

| Output | Article R2 | Article MAE | Wide R2 | Wide MAE |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

MAE is reported in each output's native units: `R` and `Z` in metres,
`rho_pol` and `CD_eta` dimensionless, and `w_cd` in normalized poloidal-radius units.

The CSV and LaTeX forms of this table are `metrics_table.csv` and
`metrics_table.tex`. Standalone and stacked parity figures, an R2 bar chart,
models, predictions, and the complete report are stored in this folder.

Reproduce with:

```bash
MPLCONFIGDIR=/tmp/mission2512_screened_mpl python mission2512_model_comparison/train_screened_models.py
```
"""
    (OUT/"README.md").write_text(readme)
    print("\nFinal test table:",flush=True)
    print("\n".join(table_rows),flush=True)


if __name__=="__main__":
    main()
