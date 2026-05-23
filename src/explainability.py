from __future__ import annotations

from typing import Dict, List

import numpy as np


def _extract_shap_vector(shap_values, class_idx: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.array(shap_values[class_idx])[0]

    arr = np.array(shap_values)
    if arr.ndim == 3:
        if arr.shape[2] > class_idx:
            return arr[0, :, class_idx]
        if arr.shape[0] > class_idx:
            return arr[class_idx, 0, :]
    if arr.ndim == 2:
        return arr[0]
    raise ValueError("Unsupported SHAP output shape")


def _top_features(values: np.ndarray, feature_names: List[str], top_k: int) -> List[Dict[str, float]]:
    abs_vals = np.abs(values)
    top_idx = np.argsort(abs_vals)[::-1][:top_k]
    return [
        {
            "feature": feature_names[i],
            "impact": float(values[i]),
            "abs_impact": float(abs_vals[i]),
        }
        for i in top_idx
    ]


class ContributionExtractor:
    def __init__(self, models: Dict[str, object], feature_names: List[str], X_background: np.ndarray):
        self.models = models
        self.feature_names = feature_names
        self.background = X_background
        self._tree_importances = {
            "random_forest": np.array(models["random_forest"].feature_importances_, dtype=float),
            "xgboost": np.array(models["xgboost"].feature_importances_, dtype=float),
        }

    def explain_row(self, x_row: np.ndarray, predicted_class_idx: int, top_k: int = 5) -> Dict[str, List[Dict[str, float]]]:
        explanations: Dict[str, List[Dict[str, float]]] = {}

        for model_name in ["random_forest", "xgboost"]:
            # Fast approximation for local evidence: feature importance weighted by row values.
            contrib = self._tree_importances[model_name] * x_row
            explanations[model_name] = _top_features(contrib, self.feature_names, top_k=top_k)

        lr_model = self.models["logistic_regression"]
        coef_vec = lr_model.coef_[predicted_class_idx]
        contrib = coef_vec * x_row
        explanations["logistic_regression"] = _top_features(contrib, self.feature_names, top_k=top_k)
        return explanations
