# tests/test_features.py
# Unit tests for the feature engineering module.
# Run with:  pytest tests/ -v

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_rolling_features,
    add_lag_features,
    add_rate_of_change,
    add_temporal_features,
    add_zone_deviation,
    engineer_features,
    get_feature_cols,
)


# ── shared fixture ─────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """A tiny but realistic-looking sensor DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "timestamp":    pd.date_range("2023-01-01", periods=n, freq="10min"),
        "sensor_id":    np.repeat(["SENSOR_001", "SENSOR_002"], n // 2),
        "zone":         np.repeat(["ZONE_A", "ZONE_B"], n // 2),
        "temperature":  np.random.normal(72, 4, n),
        "pressure":     np.random.normal(14.5, 0.5, n),
        "vibration":    np.abs(np.random.normal(0.12, 0.03, n)),
        "power_draw":   np.random.normal(220, 10, n),
        "is_anomaly":   np.zeros(n, dtype=int),
        "anomaly_type": "normal",
    })


# ── rolling features ───────────────────────────────────────────────────────
def test_rolling_adds_expected_columns(sample_df):
    df = add_rolling_features(sample_df)
    assert "temperature_roll_mean_15" in df.columns
    assert "temperature_roll_std_15"  in df.columns
    assert "power_draw_roll_max_30"   in df.columns


def test_rolling_no_new_rows(sample_df):
    df = add_rolling_features(sample_df)
    assert len(df) == len(sample_df)


def test_rolling_mean_within_reasonable_range(sample_df):
    df = add_rolling_features(sample_df)
    mean_col = df["temperature_roll_mean_15"]
    # rolling mean of temperature should stay near 72 (the true mean)
    assert mean_col.mean() == pytest.approx(72, abs=5)


# ── lag features ───────────────────────────────────────────────────────────
def test_lag_adds_expected_columns(sample_df):
    df = add_lag_features(sample_df)
    assert "temperature_lag_1" in df.columns
    assert "vibration_lag_3"   in df.columns


def test_lag_no_nan_remaining(sample_df):
    df = add_lag_features(sample_df)
    lag_cols = [c for c in df.columns if "lag_" in c]
    assert df[lag_cols].isna().sum().sum() == 0


# ── rate of change ─────────────────────────────────────────────────────────
def test_diff_columns_created(sample_df):
    df = add_rate_of_change(sample_df)
    assert "temperature_diff1" in df.columns
    assert "temperature_diff2" in df.columns


def test_first_diff_mostly_small(sample_df):
    df = add_rate_of_change(sample_df)
    # Normal data changes should be small relative to the signal
    assert df["temperature_diff1"].abs().mean() < 10


# ── temporal features ──────────────────────────────────────────────────────
def test_temporal_columns_present(sample_df):
    df = add_temporal_features(sample_df)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "is_business_hours", "is_weekend"]:
        assert col in df.columns


def test_cyclical_encoding_range(sample_df):
    df = add_temporal_features(sample_df)
    assert df["hour_sin"].between(-1, 1).all()
    assert df["hour_cos"].between(-1, 1).all()


def test_is_business_hours_binary(sample_df):
    df = add_temporal_features(sample_df)
    assert set(df["is_business_hours"].unique()).issubset({0, 1})


# ── zone deviation ─────────────────────────────────────────────────────────
def test_zone_zscore_columns_created(sample_df):
    df = add_zone_deviation(sample_df)
    assert "temperature_zone_zscore" in df.columns
    assert "power_draw_zone_zscore"  in df.columns


# ── full pipeline ──────────────────────────────────────────────────────────
def test_engineer_features_no_nan(sample_df):
    df = engineer_features(sample_df)
    assert df.isna().sum().sum() == 0


def test_engineer_features_row_count(sample_df):
    df = engineer_features(sample_df)
    assert len(df) == len(sample_df)


def test_get_feature_cols_excludes_metadata(sample_df):
    df = engineer_features(sample_df)
    feat_cols = get_feature_cols(df)
    for excluded in ["timestamp", "sensor_id", "zone", "is_anomaly", "anomaly_type"]:
        assert excluded not in feat_cols


def test_feature_count_reasonable(sample_df):
    df = engineer_features(sample_df)
    # With 4 sensor cols × (4 rolling × 3 windows + 3 lags + 2 diffs + 1 zscore)
    # plus 7 temporal cols = should be well above 40 features
    assert len(get_feature_cols(df)) > 40
