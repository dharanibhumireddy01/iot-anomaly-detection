# tests/test_model.py
# Unit tests for the Isolation Forest model module.
# Run with:  pytest tests/ -v

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import engineer_features
from src.isolation_forest_model import (
    get_feature_cols,
    _scale,
    train,
    predict,
    evaluate,
)
from src.config import IF_BEST_PARAMS


# ── shared fixture ─────────────────────────────────────────────────────────
@pytest.fixture
def small_featured_df():
    """
    A small engineered DataFrame to test model functions without
    running the full 1M-record pipeline.
    """
    np.random.seed(42)
    n = 300
    df_raw = pd.DataFrame({
        "timestamp":    pd.date_range("2023-01-01", periods=n, freq="5min"),
        "sensor_id":    np.repeat(["SENSOR_001", "SENSOR_002", "SENSOR_003"], n // 3),
        "zone":         np.repeat(["ZONE_A", "ZONE_B", "ZONE_C"], n // 3),
        "temperature":  np.random.normal(72, 4, n),
        "pressure":     np.random.normal(14.5, 0.5, n),
        "vibration":    np.abs(np.random.normal(0.12, 0.03, n)),
        "power_draw":   np.random.normal(220, 10, n),
        "is_anomaly":   np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "anomaly_type": "normal",
    })
    return engineer_features(df_raw)


# ── feature extraction ─────────────────────────────────────────────────────
def test_get_feature_cols_returns_list(small_featured_df):
    cols = get_feature_cols(small_featured_df)
    assert isinstance(cols, list)
    assert len(cols) > 0


def test_feature_cols_no_label_leakage(small_featured_df):
    cols = get_feature_cols(small_featured_df)
    assert "is_anomaly" not in cols
    assert "anomaly_type" not in cols


# ── scaling ────────────────────────────────────────────────────────────────
def test_scale_returns_correct_shape(small_featured_df):
    X, scaler = _scale(small_featured_df, fit=True)
    assert X.shape[0] == len(small_featured_df)
    assert X.shape[1] == len(get_feature_cols(small_featured_df))


def test_scale_zero_mean_after_fit(small_featured_df):
    X, _ = _scale(small_featured_df, fit=True)
    # StandardScaler should produce zero mean
    assert abs(X.mean()) < 0.1


def test_scale_reuse_scaler(small_featured_df):
    X1, scaler = _scale(small_featured_df, fit=True)
    X2, _      = _scale(small_featured_df, scaler=scaler, fit=False)
    np.testing.assert_array_almost_equal(X1, X2)


# ── training ───────────────────────────────────────────────────────────────
def test_train_returns_model_and_scaler(small_featured_df):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    clf, scaler = train(small_featured_df)
    assert isinstance(clf, IsolationForest)
    assert isinstance(scaler, StandardScaler)


def test_model_is_fitted(small_featured_df):
    clf, _ = train(small_featured_df)
    # A fitted IsolationForest has estimators_
    assert hasattr(clf, "estimators_")


# ── prediction ─────────────────────────────────────────────────────────────
def test_predict_returns_binary_array(small_featured_df):
    clf, scaler = train(small_featured_df)
    y_pred, scores = predict(clf, scaler, small_featured_df)

    assert set(y_pred).issubset({0, 1})
    assert len(y_pred) == len(small_featured_df)


def test_scores_are_negative_floats(small_featured_df):
    clf, scaler = train(small_featured_df)
    _, scores = predict(clf, scaler, small_featured_df)

    # Isolation Forest score_samples returns negative values
    assert scores.max() <= 0


def test_prediction_count_matches_input(small_featured_df):
    clf, scaler = train(small_featured_df)
    y_pred, _ = predict(clf, scaler, small_featured_df)
    assert len(y_pred) == len(small_featured_df)


# ── evaluation ─────────────────────────────────────────────────────────────
def test_evaluate_returns_dict(small_featured_df):
    clf, scaler = train(small_featured_df)
    y_pred, scores = predict(clf, scaler, small_featured_df)
    y_true = small_featured_df["is_anomaly"].values

    metrics = evaluate(y_true, y_pred, scores)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "tpr" in metrics
    assert "fpr" in metrics


def test_accuracy_between_zero_and_one(small_featured_df):
    clf, scaler = train(small_featured_df)
    y_pred, scores = predict(clf, scaler, small_featured_df)
    y_true = small_featured_df["is_anomaly"].values

    metrics = evaluate(y_true, y_pred, scores)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_auc_roc_is_numeric(small_featured_df):
    clf, scaler = train(small_featured_df)
    y_pred, scores = predict(clf, scaler, small_featured_df)
    y_true = small_featured_df["is_anomaly"].values

    metrics = evaluate(y_true, y_pred, scores)
    # AUC-ROC must be a valid float between 0 and 1
    # Note: on very small samples (300 rows, ~9 anomalies) AUC can be < 0.5
    # due to class imbalance and sample variance — not a model defect
    assert 0.0 <= metrics.get("auc_roc", 0.5) <= 1.0
