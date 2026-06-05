"""Feature engineering pipeline for IBM HR employee attrition."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = Path("data/raw/hr_attrition.csv")
PROCESSED_PATH = Path("data/processed/attrition_features.csv")
TRAIN_PATH = Path("data/processed/train_features.csv")
TEST_PATH = Path("data/processed/test_features.csv")
REFERENCE_PATH = Path("data/reference/reference_dataset.csv")
SCALER_PATH = Path("models/scaler.joblib")
PREPROCESSING_PATH = Path("models/preprocessing.joblib")

TARGET_COLUMN = "Attrition"
ID_COLUMN = "EmployeeNumber"
DROP_CONSTANT_COLUMNS = ["EmployeeCount", "StandardHours", "Over18"]
EXCLUDED_FEATURE_COLUMNS = [ID_COLUMN, *DROP_CONSTANT_COLUMNS]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw HR attrition dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path)


def encode_target(series: pd.Series) -> pd.Series:
    """Map the Yes/No attrition target to binary labels."""
    encoded = series.map({"Yes": 1, "No": 0})
    if encoded.isna().any():
        bad_values = sorted(series.loc[encoded.isna()].dropna().astype(str).unique())
        raise ValueError(f"Unexpected Attrition values: {bad_values}")
    return encoded.astype(int)


def feature_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Return raw input feature columns and typed feature groups."""
    missing = [column for column in [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN and column not in EXCLUDED_FEATURE_COLUMNS
    ]
    categorical_columns = [
        column
        for column in raw_feature_columns
        if df[column].dtype == "object" or str(df[column].dtype).startswith("category")
    ]
    numeric_columns = [column for column in raw_feature_columns if column not in categorical_columns]
    return {
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "raw_feature_columns": raw_feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "excluded_features": [column for column in EXCLUDED_FEATURE_COLUMNS if column in df.columns],
    }


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a fitted-on-train-only preprocessing object."""
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split raw data before fitting any preprocessing."""
    y = encode_target(df[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN])
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names as plain strings."""
    return [str(name) for name in preprocessor.get_feature_names_out()]


def transform_to_frame(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Transform a raw feature frame into a model-ready DataFrame."""
    names = feature_names or transformed_feature_names(preprocessor)
    transformed = preprocessor.transform(frame)
    return pd.DataFrame(transformed, columns=names, index=frame.index)


def engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split raw data, fit preprocessing on train only, and transform datasets."""
    schema = feature_schema(df)
    X_train, X_test, y_train, y_test = split_raw_data(df)
    raw_feature_columns = schema["raw_feature_columns"]

    preprocessor = build_preprocessor(
        schema["numeric_columns"],
        schema["categorical_columns"],
        scale_numeric=True,
    )
    preprocessor.fit(X_train[raw_feature_columns])
    feature_names = transformed_feature_names(preprocessor)

    train_features = transform_to_frame(preprocessor, X_train[raw_feature_columns], feature_names)
    test_features = transform_to_frame(preprocessor, X_test[raw_feature_columns], feature_names)
    full_features = transform_to_frame(preprocessor, df[raw_feature_columns], feature_names)

    train_scaled = train_features.copy()
    train_scaled[TARGET_COLUMN] = y_train.values
    test_scaled = test_features.copy()
    test_scaled[TARGET_COLUMN] = y_test.values
    full_scaled = full_features.copy()
    full_scaled[TARGET_COLUMN] = encode_target(df[TARGET_COLUMN]).values

    if ID_COLUMN in df.columns:
        train_scaled[ID_COLUMN] = X_train[ID_COLUMN].values
        test_scaled[ID_COLUMN] = X_test[ID_COLUMN].values
        full_scaled[ID_COLUMN] = df[ID_COLUMN].values

    metadata = {
        **schema,
        "feature_columns": feature_names,
        "preprocessor": preprocessor,
        "scale_numeric": True,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "risk_bands": {
            "low_threshold": 0.35,
            "medium_threshold": 0.5,
            "high_threshold": 0.65,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    reference = train_scaled.copy()
    return full_scaled, train_scaled, test_scaled, reference, metadata


def save_feature_outputs(
    full_scaled: pd.DataFrame,
    train_scaled: pd.DataFrame,
    test_scaled: pd.DataFrame,
    reference: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Persist processed features, reference data, scaler, and preprocessing metadata."""
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    full_scaled.to_csv(PROCESSED_PATH, index=False)
    train_scaled.to_csv(TRAIN_PATH, index=False)
    test_scaled.to_csv(TEST_PATH, index=False)
    reference.to_csv(REFERENCE_PATH, index=False)

    preprocessor = metadata["preprocessor"]
    try:
        scaler = preprocessor.named_transformers_["numeric"].named_steps.get("scaler")
        if scaler is not None:
            joblib.dump(scaler, SCALER_PATH)
    except Exception:
        pass
    joblib.dump(metadata, PREPROCESSING_PATH)


def main() -> dict[str, Any]:
    """Run the full feature engineering pipeline."""
    raw_df = load_raw_data()
    full_scaled, train_scaled, test_scaled, reference, metadata = engineer_features(raw_df)
    save_feature_outputs(full_scaled, train_scaled, test_scaled, reference, metadata)
    summary = {
        "processed_path": str(PROCESSED_PATH),
        "train_rows": int(train_scaled.shape[0]),
        "test_rows": int(test_scaled.shape[0]),
        "feature_count": len(metadata["feature_columns"]),
        "excluded_features": metadata["excluded_features"],
        "scaler_path": str(SCALER_PATH),
        "preprocessing_path": str(PREPROCESSING_PATH),
        "reference_path": str(REFERENCE_PATH),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
