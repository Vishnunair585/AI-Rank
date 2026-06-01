"""
utils/ml_model.py
Trains a Linear Regression model to predict AI tool suitability scores
per task category. Also provides ranking utilities.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.data_loader import (
    TASK_COLUMNS, FEATURE_COLUMNS,
    get_features_and_labels, load_data
)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_model(task: str) -> dict:
    """
    Train a Linear Regression model for a specific task.

    Returns a dict with:
        model, scaler, r2, mae, rmse, cv_scores
    """
    X, y, names = get_features_and_labels(task)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = cross_val_score(model, X_scaled, y, cv=3, scoring="r2")

    # Save
    joblib.dump(model,  MODELS_DIR / f"{task}_model.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{task}_scaler.pkl")

    return {
        "model":     model,
        "scaler":    scaler,
        "r2":        round(r2, 4),
        "mae":       round(mae, 4),
        "rmse":      round(rmse, 4),
        "cv_scores": cv.tolist(),
        "cv_mean":   round(cv.mean(), 4),
    }


def train_all_models() -> dict:
    """Train and save models for all 25 task categories."""
    results = {}
    for task in TASK_COLUMNS:
        print(f"  Training model for: {task}")
        results[task] = train_model(task)
    print(f"\nAll {len(TASK_COLUMNS)} models trained and saved.")
    return results


def load_model(task: str):
    """Load a saved model and scaler for a task. Train on-the-fly if missing."""
    model_path  = MODELS_DIR / f"{task}_model.pkl"
    scaler_path = MODELS_DIR / f"{task}_scaler.pkl"

    if not model_path.exists():
        train_model(task)

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_scores(task: str) -> pd.DataFrame:
    """
    Predict suitability scores for all AI tools for a given task.

    Returns a DataFrame sorted by predicted score (descending):
        tool_name | predicted_score | rank | actual_score
    """
    model, scaler = load_model(task)
    X, y, names   = get_features_and_labels(task)

    X_scaled        = scaler.transform(X)
    predicted       = model.predict(X_scaled)
    predicted_clipped = np.clip(predicted, 0, 100)

    result = pd.DataFrame({
        "tool_name":       names.values,
        "predicted_score": np.round(predicted_clipped, 1),
        "actual_score":    y.values,
    })

    result = result.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


def get_top_tool(task: str) -> dict:
    """Return the top-ranked tool and its score for a task."""
    df    = predict_scores(task)
    top   = df.iloc[0]
    return {
        "tool_name":       top["tool_name"],
        "predicted_score": top["predicted_score"],
        "rank":            1
    }


def get_model_metrics(task: str) -> dict:
    """Return evaluation metrics for a task's model."""
    metrics = train_model(task)
    return {k: v for k, v in metrics.items() if k != "model" and k != "scaler"}


def get_feature_importance(task: str) -> pd.DataFrame:
    """Return feature coefficients from the linear regression model."""
    model, _ = load_model(task)
    importance = pd.DataFrame({
        "feature":     FEATURE_COLUMNS,
        "coefficient": np.round(model.coef_, 4)
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)
    return importance
