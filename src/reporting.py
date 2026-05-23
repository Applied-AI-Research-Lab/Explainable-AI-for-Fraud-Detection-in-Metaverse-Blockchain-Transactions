from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .data_utils import LABEL_ORDER


def ensure_dirs(base_dir: str) -> Dict[str, str]:
    paths = {
        "base": base_dir,
        "metrics": os.path.join(base_dir, "metrics"),
        "figures": os.path.join(base_dir, "figures"),
        "tables": os.path.join(base_dir, "tables"),
        "predictions": os.path.join(base_dir, "predictions"),
        "prompts": os.path.join(base_dir, "prompts"),
        "models": os.path.join(base_dir, "models"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_metrics_tables(paths: Dict[str, str], metrics_rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(metrics_rows)
    csv_path = os.path.join(paths["metrics"], "model_metrics.csv")
    tex_path = os.path.join(paths["tables"], "model_metrics.tex")
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format="%.4f")
    return df


def save_confusion_figure(
    paths: Dict[str, str],
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(paths["figures"], f"cm_{model_name}.png"), dpi=320)
    plt.close(fig)


def save_metric_barplot(paths: Dict[str, str], metrics_df: pd.DataFrame) -> None:
    plot_df = metrics_df[["model", "accuracy", "f1_macro", "balanced_accuracy"]].melt(
        id_vars="model", var_name="metric", value_name="value"
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sns.barplot(data=plot_df, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(os.path.join(paths["figures"], "metrics_barplot.png"), dpi=320)
    plt.close(fig)


def save_agreement_tables(paths: Dict[str, str], agreement_stats: Dict[str, float]) -> None:
    df = pd.DataFrame([agreement_stats])
    df.to_csv(os.path.join(paths["metrics"], "agreement_stats.csv"), index=False)
    df.to_latex(
        os.path.join(paths["tables"], "agreement_stats.tex"),
        index=False,
        float_format="%.4f",
    )


def save_class_distribution_table(paths: Dict[str, str], labels: np.ndarray) -> None:
    df = pd.Series(labels).value_counts().rename_axis("class").reset_index(name="count")
    total = df["count"].sum()
    df["ratio"] = df["count"] / total
    df.to_csv(os.path.join(paths["metrics"], "test_class_distribution.csv"), index=False)
    df.to_latex(
        os.path.join(paths["tables"], "test_class_distribution.tex"),
        index=False,
        float_format="%.4f",
    )
