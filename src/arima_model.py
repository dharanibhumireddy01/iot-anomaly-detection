# arima_model.py
# ARIMA-based statistical baseline for anomaly detection.
#
# The idea: fit a time-series model on each sensor's historical readings
# to predict what the next reading "should" be. If the actual reading
# is very far from the prediction (> 3 standard deviations), flag it.
#
# This is the statistical layer. It's good at catching smooth anomalies
# like drift and some spikes, but struggles with sudden pattern changes.
# That's why we also run Isolation Forest on top of this.
#
# ARIMA(2, 1, 2) means:
#   p=2: use last 2 values to predict next (autoregression)
#   d=1: difference the series once to remove trend
#   q=2: use last 2 prediction errors in the model (moving average)

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.config import (
    ARIMA_ORDER, ARIMA_TARGET_COL, ARIMA_RESID_SIGMA, ARIMA_MIN_ROWS
)
from src.logger import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore")


def _is_stationary(series: pd.Series):
    """
    Run the Augmented Dickey-Fuller test to check if the series is stationary.
    Returns (is_stationary, p_value).

    Stationary means the statistical properties don't change over time —
    required before fitting ARIMA. If not stationary, we difference it (d=1).
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1] < 0.05, round(result[1], 4)


def _fit_one_sensor(series: pd.Series):
    """
    Fits ARIMA on a single sensor's readings.
    Auto-detects whether differencing is needed based on ADF test.
    Returns a dict with the fitted model results, or None if fitting fails.
    """
    series = series.dropna().reset_index(drop=True)

    if len(series) < ARIMA_MIN_ROWS:
        return None

    is_stat, _ = _is_stationary(series)
    d = 0 if is_stat else 1
    order = (ARIMA_ORDER[0], d, ARIMA_ORDER[2])

    try:
        model  = ARIMA(series, order=order)
        result = model.fit()

        fitted    = result.fittedvalues
        residuals = series - fitted
        threshold = ARIMA_RESID_SIGMA * residuals.std()

        return {
            "fitted":     fitted,
            "residuals":  residuals,
            "threshold":  threshold,
            "aic":        round(result.aic, 2),
            "order":      order,
        }
    except Exception as e:
        log.warning(f"ARIMA fitting failed: {e}")
        return None


def run_arima(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits ARIMA per sensor and flags records where the residual
    exceeds ARIMA_RESID_SIGMA standard deviations from zero.

    Adds three columns to the DataFrame:
        arima_fitted        — what ARIMA predicted
        arima_residual      — actual minus predicted
        arima_anomaly_flag  — 1 if abs(residual) > threshold, else 0
    """
    df = df.copy()
    df["arima_fitted"]       = df[ARIMA_TARGET_COL].values
    df["arima_residual"]     = 0.0
    df["arima_anomaly_flag"] = 0

    sensors = df["sensor_id"].unique()
    total_flagged = 0
    skipped = 0

    log.info(f"Fitting ARIMA{ARIMA_ORDER} on '{ARIMA_TARGET_COL}' for {len(sensors)} sensors")

    for i, sid in enumerate(sensors, 1):
        mask   = df["sensor_id"] == sid
        series = df.loc[mask, ARIMA_TARGET_COL].reset_index(drop=True)

        result = _fit_one_sensor(series)

        if result is None:
            skipped += 1
            continue

        idx = df.index[mask][: len(result["residuals"])]
        df.loc[idx, "arima_fitted"]       = result["fitted"].values
        df.loc[idx, "arima_residual"]     = result["residuals"].values
        df.loc[idx, "arima_anomaly_flag"] = (
            np.abs(result["residuals"].values) > result["threshold"]
        ).astype(int)

        total_flagged += df.loc[idx, "arima_anomaly_flag"].sum()

        if i % 10 == 0:
            log.info(f"  {i}/{len(sensors)} sensors done | flagged so far: {total_flagged:,}")

    log.info(f"ARIMA complete — flagged {total_flagged:,} records, skipped {skipped} sensors")
    return df


def evaluate_arima(df: pd.DataFrame) -> None:
    """Quick precision/recall check against ground-truth labels."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = df["is_anomaly"].values
    y_pred = df["arima_anomaly_flag"].values

    print("\n── ARIMA Evaluation ──")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"False Positive Rate: {fp / (fp + tn):.4f}")
    print(f"True Positive Rate : {tp / (tp + fn):.4f}")


if __name__ == "__main__":
    df = pd.read_csv("data/iot_sensor_data.csv", nrows=50_000)
    df = run_arima(df)
    evaluate_arima(df)
