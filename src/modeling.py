from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier


def build_models(random_state: int, n_classes: int) -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=30,
            class_weight="balanced_subsample",
            max_depth=10,
            random_state=random_state,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=30,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def train_models(
    models: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, object]:
    fitted = {}
    for name, model in models.items():
        fitted[name] = model.fit(X_train, y_train)
    return fitted


def predict_models(
    models: Dict[str, object],
    X_test: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        output[name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    return output


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics, report
