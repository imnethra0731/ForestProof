"""Small, auditable ML service for ForestProof.

The model is intentionally trained only from user-supplied labelled data. The API
never invents accuracy numbers or silently treats NDVI rules as ML predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

MODEL_PATH = Path(__file__).resolve().parent / "forestproof_outputs" / "activity_model.joblib"
MODEL_PATH.parent.mkdir(exist_ok=True)

FEATURES = [
    "ndvi_before",
    "ndvi_after",
    "ndvi_drop",
    "red_before",
    "nir_before",
    "red_after",
    "nir_after",
    "valid_coverage_percent",
    "candidate_area_hectares",
]


def _rows_to_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) < 10:
        raise ValueError("At least 10 labelled rows are required for ML training.")
    missing = [f for f in FEATURES if any(f not in row for row in rows)]
    if missing:
        raise ValueError(f"Training data is missing feature columns: {', '.join(missing)}")
    labels = [str(row.get("label", "")).strip() for row in rows]
    if any(not label for label in labels):
        raise ValueError("Every training row needs a non-empty label column.")
    if len(set(labels)) < 2:
        raise ValueError("Training data must contain at least two different labels.")
    X = np.asarray([[float(row[f]) for f in FEATURES] for row in rows], dtype=np.float32)
    y = np.asarray(labels)
    return X, y


def train(rows: list[dict[str, Any]], test_size: float = 0.25, random_state: int = 42) -> dict[str, Any]:
    X, y = _rows_to_xy(rows)
    counts = {label: int((y == label).sum()) for label in sorted(set(y))}
    if min(counts.values()) < 2:
        raise ValueError("Each class needs at least two examples for a stratified validation split.")
    groups = np.asarray([str(row.get("group", "")).strip() for row in rows])
    grouped_validation = bool(np.all(groups) and len(set(groups)) >= 4)
    if grouped_validation:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            raise ValueError("Grouped validation split must contain at least two classes in both train and test groups.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=random_state,
        min_samples_leaf=2, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    labels = sorted(set(y))
    report = classification_report(y_test, predicted, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "macro_precision": round(float(precision_score(y_test, predicted, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_test, predicted, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro", zero_division=0)), 4),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "class_labels": labels,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_counts": counts,
        "features": FEATURES,
        "validation_method": "grouped_holdout" if grouped_validation else "stratified_random_holdout",
        "groups_used": sorted(set(groups)) if grouped_validation else None,
        "warning": "Validation is only as representative as the supplied labels; it is not field-truth accuracy.",
    }
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    return metrics


def predict(features: dict[str, Any]) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model exists. Upload labelled training data to /train-ml first.")
    bundle = joblib.load(MODEL_PATH)
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Prediction is missing feature values: {', '.join(missing)}")
    X = np.asarray([[float(features[f]) for f in FEATURES]], dtype=np.float32)
    model = bundle["model"]
    label = str(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    return {
        "label": label,
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(model.classes_, probabilities)},
        "features": FEATURES,
        "validation": bundle.get("metrics"),
        "warning": "This is a model prediction for review, not proof of an activity or illegality.",
    }


def model_status() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {"trained": False, "model_path": str(MODEL_PATH), "features": FEATURES}
    bundle = joblib.load(MODEL_PATH)
    return {"trained": True, "model_path": str(MODEL_PATH), "features": FEATURES, "validation": bundle.get("metrics")}
