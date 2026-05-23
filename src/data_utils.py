from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "anomaly"
LABEL_ORDER = ["low_risk", "moderate_risk", "high_risk"]


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    y_test_labels: np.ndarray


def load_dataset(path: str, max_rows: int | None = None) -> pd.DataFrame:
    if max_rows is not None and max_rows > 0:
        df = pd.read_csv(path, nrows=max_rows)
    else:
        df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "timestamp" in result.columns:
        ts = pd.to_datetime(result["timestamp"], errors="coerce")
        result["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
        result["month"] = ts.dt.month.fillna(1).astype(int)
        result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    else:
        result["day_of_week"] = 0
        result["month"] = 1
        result["is_weekend"] = 0
    return result


def _fit_frequency_map(series: pd.Series) -> Dict[str, float]:
    counts = series.astype(str).value_counts(normalize=True)
    return counts.to_dict()


def _apply_frequency_map(series: pd.Series, mapping: Dict[str, float]) -> pd.Series:
    return series.astype(str).map(mapping).fillna(0.0)


def _prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_risk_score: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    train_feat = _add_time_features(train_df)
    test_feat = _add_time_features(test_df)

    for col in ["sending_address", "receiving_address"]:
        if col in train_feat.columns and col in test_feat.columns:
            freq_map = _fit_frequency_map(train_feat[col])
            train_feat[f"{col}_freq"] = _apply_frequency_map(train_feat[col], freq_map)
            test_feat[f"{col}_freq"] = _apply_frequency_map(test_feat[col], freq_map)

    drop_cols = [TARGET_COLUMN, "timestamp", "sending_address", "receiving_address"]
    if not include_risk_score and "risk_score" in train_feat.columns:
        drop_cols.append("risk_score")

    X_train_df = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns]).copy()
    X_test_df = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns]).copy()

    high_cardinality_cols = []
    for col in X_train_df.columns:
        if not is_numeric_dtype(X_train_df[col]):
            if X_train_df[col].nunique(dropna=True) > 100:
                high_cardinality_cols.append(col)

    for col in high_cardinality_cols:
        freq_map = _fit_frequency_map(X_train_df[col])
        X_train_df[f"{col}_freq"] = _apply_frequency_map(X_train_df[col], freq_map)
        X_test_df[f"{col}_freq"] = _apply_frequency_map(X_test_df[col], freq_map)
        X_train_df = X_train_df.drop(columns=[col])
        X_test_df = X_test_df.drop(columns=[col])

    if "amount" in X_train_df.columns:
        X_train_df["amount_log1p"] = np.log1p(X_train_df["amount"].clip(lower=0))
        X_test_df["amount_log1p"] = np.log1p(X_test_df["amount"].clip(lower=0))

    # Treat only true numeric dtypes as numerical; everything else is categorical.
    # This prevents string-like columns (e.g., pandas string/category) from reaching StandardScaler.
    numerical_cols = [c for c in X_train_df.columns if is_numeric_dtype(X_train_df[c])]
    categorical_cols = [c for c in X_train_df.columns if c not in numerical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    feature_names = preprocessor.get_feature_names_out().tolist()
    return X_train, X_test, feature_names


def _encode_labels(y: pd.Series) -> np.ndarray:
    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    return y.map(label_to_id).to_numpy(dtype=int)


def decode_labels(y_encoded: np.ndarray) -> np.ndarray:
    id_to_label = {idx: label for idx, label in enumerate(LABEL_ORDER)}
    return np.array([id_to_label[int(v)] for v in y_encoded])


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    include_risk_score: bool = False,
) -> PreparedData:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    X_train, X_test, feature_names = _prepare_features(
        train_df=train_df,
        test_df=test_df,
        include_risk_score=include_risk_score,
    )

    y_train = _encode_labels(train_df[TARGET_COLUMN])
    y_test = _encode_labels(test_df[TARGET_COLUMN])

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        train_raw=train_df.reset_index(drop=True),
        test_raw=test_df.reset_index(drop=True),
        y_test_labels=test_df[TARGET_COLUMN].to_numpy(),
    )
