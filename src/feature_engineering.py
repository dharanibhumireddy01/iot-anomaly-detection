# feature_engineering.py
# Takes raw sensor readings and builds the feature matrix
# that the ML model actually trains on.
#
# The raw columns (temperature, pressure, vibration, power_draw) alone
# are not enough to detect anomalies reliably. What matters is:
#   - How does this reading compare to the last 5/15/30 readings?
#   - Is the rate of change suddenly different?
#   - Is this sensor reading unusually different from others in its zone?
#   - Is this happening at a suspicious time of day?
#
# All of that context gets captured here as engineered features.

import pandas as pd
import numpy as np
from typing import List

from src.config import SENSOR_COLS, ROLL_WINDOWS, LAG_STEPS
from src.logger import get_logger

log = get_logger(__name__)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sensor column, compute rolling mean, std, min, and max
    over windows of 5, 15, and 30 readings.

    Rolling features answer the question:
    "Is this reading normal given recent history?"
    """
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    for col in SENSOR_COLS:
        grouped = df.groupby("sensor_id")[col]
        for w in ROLL_WINDOWS:
            df[f"{col}_roll_mean_{w}"] = grouped.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{w}"] = grouped.transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
            df[f"{col}_roll_max_{w}"] = grouped.transform(
                lambda x: x.rolling(w, min_periods=1).max()
            )
            df[f"{col}_roll_min_{w}"] = grouped.transform(
                lambda x: x.rolling(w, min_periods=1).min()
            )

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previous values for each sensor reading at t-1, t-2, t-3.

    Lag features answer the question:
    "Is this reading very different from what this sensor just reported?"
    This is especially useful for catching sudden spikes and flatlines.
    """
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    for col in SENSOR_COLS:
        grouped = df.groupby("sensor_id")[col]
        for lag in LAG_STEPS:
            df[f"{col}_lag_{lag}"] = grouped.shift(lag).bfill()

    return df


def add_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    First-order diff: how much did the reading change since last time?
    Second-order diff: is the rate of change itself accelerating?

    These two features together are very good at catching:
    - Spikes (huge first-order jump)
    - Drift (small first-order, but consistent — shows up as rising second-order)
    """
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)

    for col in SENSOR_COLS:
        grouped = df.groupby("sensor_id")[col]
        diff1 = grouped.diff().fillna(0)
        diff2 = diff1.groupby(df["sensor_id"]).diff().fillna(0)

        df[f"{col}_diff1"] = diff1
        df[f"{col}_diff2"] = diff2

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode time of day and day of week as sine/cosine pairs.

    Why sin/cos and not just the raw hour?
    Because hour 23 and hour 0 are actually 1 hour apart, but
    numerically they're 23 apart. The circular encoding fixes that.

    is_business_hours flags 8am-6pm activity as normal,
    everything else as potentially suspicious.
    """
    ts  = pd.to_datetime(df["timestamp"])
    hr  = ts.dt.hour
    dow = ts.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hr  / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hr  / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

    df["is_business_hours"] = ((hr >= 8) & (hr <= 18)).astype(int)
    df["is_weekend"]        = (dow >= 5).astype(int)
    df["hour_raw"]          = hr    # keep raw too for the zone heatmap viz

    return df


def add_zone_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    How far is this sensor's reading from the median of all sensors in its zone?
    Expressed as a z-score.

    A sensor that's running at 2× the temperature of every other sensor
    in ZONE_A is suspicious even if its absolute temperature looks normal.
    This feature captures cross-sensor abnormality.
    """
    for col in SENSOR_COLS:
        zone_stats = (
            df.groupby("zone")[col]
            .agg(zone_median="median", zone_std="std")
            .reset_index()
        )
        zone_stats["zone_std"] = zone_stats["zone_std"].replace(0, 1)

        df = df.merge(zone_stats, on="zone", how="left")

        df[f"{col}_zone_zscore"] = (
            (df[col] - df["zone_median"]) / df["zone_std"]
        )

        df.drop(columns=["zone_median", "zone_std"], inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all feature engineering steps in order.
    Returns the full feature matrix ready for model training.
    """
    log.info(f"Starting feature engineering — input shape: {df.shape}")

    log.info("  Step 1/5 — Rolling statistics (windows: 5, 15, 30)")
    df = add_rolling_features(df)

    log.info("  Step 2/5 — Lag features (t-1, t-2, t-3)")
    df = add_lag_features(df)

    log.info("  Step 3/5 — Rate-of-change (diff1, diff2)")
    df = add_rate_of_change(df)

    log.info("  Step 4/5 — Temporal encoding (hour, day-of-week)")
    df = add_temporal_features(df)

    log.info("  Step 5/5 — Zone deviation z-scores")
    df = add_zone_deviation(df)

    # Fill any remaining NaN from edge cases at the start of each sensor stream
    df = df.fillna(0)

    n_feat = len(get_feature_cols(df))
    log.info(f"Feature engineering complete — output shape: {df.shape}, features: {n_feat}")

    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Returns only the engineered feature column names.
    Excludes metadata, labels, and anything the model shouldn't see.
    """
    exclude = {
        "timestamp", "sensor_id", "zone",
        "is_anomaly", "anomaly_type",
        "arima_residual", "arima_fitted", "arima_anomaly_flag",
        "hour_raw",
    }
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    df_raw = pd.read_csv("data/iot_sensor_data.csv", nrows=10_000)
    df_feat = engineer_features(df_raw)
    print(f"\nFeature columns ({len(get_feature_cols(df_feat))}):")
    print(get_feature_cols(df_feat))
