# alerting.py
# Reads model predictions and generates structured alert reports.
#
# In a real deployment this module would push alerts to:
#   - A monitoring dashboard (Grafana, CloudWatch, Power BI streaming)
#   - An email/Slack notification system
#   - A SIEM (Security Information and Event Management) platform
#
# For now it writes clean CSV and text reports to outputs/alerts/
# so the pipeline is end-to-end and the claims in the resume are accurate.
#
# Three alert levels:
#   CRITICAL — anomaly score very low (most confident anomaly)
#   WARNING  — anomaly score moderately low
#   CLUSTER  — 5+ anomalies from the same sensor in the last hour (burst)

import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import (
    CRITICAL_SCORE_THRESHOLD,
    WARNING_SCORE_THRESHOLD,
    CLUSTER_THRESHOLD,
    ALERTS_DIR,
    ALERT_REPORT_PATH,
)
from src.logger import get_logger

log = get_logger(__name__)


def _assign_severity(score: float) -> str:
    """
    Assigns CRITICAL / WARNING / INFO based on the anomaly score.
    Isolation Forest scores are negative — lower = more anomalous.
    """
    if score <= CRITICAL_SCORE_THRESHOLD:
        return "CRITICAL"
    elif score <= WARNING_SCORE_THRESHOLD:
        return "WARNING"
    else:
        return "INFO"


def _find_clusters(df_anomalies: pd.DataFrame) -> pd.DataFrame:
    """
    Looks for sensors that generate CLUSTER_THRESHOLD or more anomalies
    within any 1-hour window. This is a strong signal for a compromised
    or malfunctioning sensor that needs immediate investigation.
    """
    if df_anomalies.empty:
        return pd.DataFrame()

    df = df_anomalies.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "timestamp"])
    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    cluster_counts = (
        df.groupby(["sensor_id", "zone", "hour_bucket"])
        .size()
        .reset_index(name="anomaly_count")
    )

    clusters = cluster_counts[cluster_counts["anomaly_count"] >= CLUSTER_THRESHOLD].copy()
    clusters["alert_type"] = "CLUSTER"
    clusters["detail"] = (
        clusters["anomaly_count"].astype(str)
        + " anomalies from "
        + clusters["sensor_id"]
        + " in 1 hour"
    )

    return clusters


def generate_alert_report(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the full predictions DataFrame and produces an alert report.

    Returns a DataFrame of alerts sorted by severity.
    Also saves two files:
        outputs/alerts/alert_report.csv   — machine-readable
        outputs/alerts/alert_summary.txt  — human-readable summary
    """
    os.makedirs(ALERTS_DIR, exist_ok=True)

    # ── filter to predicted anomalies ─────────────────────────────────────
    df_anom = df_predictions[df_predictions["if_pred"] == 1].copy()

    if df_anom.empty:
        log.warning("No anomalies detected — nothing to alert on")
        return pd.DataFrame()

    log.info(f"Generating alerts for {len(df_anom):,} predicted anomalies")

    # ── assign severity to each anomaly ───────────────────────────────────
    df_anom["severity"] = df_anom["if_score"].apply(_assign_severity)

    # ── build base alert records ───────────────────────────────────────────
    alerts = df_anom[[
        "timestamp", "sensor_id", "zone",
        "temperature", "pressure", "vibration", "power_draw",
        "if_score", "severity",
        "is_anomaly",       # ground truth (in production this would not exist)
        "anomaly_type",     # ground truth
    ]].copy()

    alerts["alert_type"]    = "ANOMALY"
    alerts["generated_at"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── find burst clusters ────────────────────────────────────────────────
    clusters = _find_clusters(df_anom)
    n_clusters = len(clusters)

    # ── severity breakdown ─────────────────────────────────────────────────
    severity_counts = alerts["severity"].value_counts().to_dict()
    n_critical = severity_counts.get("CRITICAL", 0)
    n_warning  = severity_counts.get("WARNING",  0)
    n_info     = severity_counts.get("INFO",     0)

    # ── most affected zones ────────────────────────────────────────────────
    top_zones = (
        alerts.groupby("zone")["sensor_id"].count()
        .sort_values(ascending=False)
        .head(3)
    )

    # ── most affected sensors ──────────────────────────────────────────────
    top_sensors = (
        alerts.groupby("sensor_id")["if_score"].count()
        .sort_values(ascending=False)
        .head(5)
    )

    # ── off-hours anomalies ────────────────────────────────────────────────
    alerts["hour"] = pd.to_datetime(alerts["timestamp"]).dt.hour
    off_hours_pct  = ((alerts["hour"] < 8) | (alerts["hour"] > 18)).mean() * 100

    # ── save CSV report ────────────────────────────────────────────────────
    alerts_sorted = alerts.sort_values(
        ["severity", "if_score"],
        key=lambda col: col.map({"CRITICAL": 0, "WARNING": 1, "INFO": 2})
        if col.name == "severity" else col
    )
    alerts_sorted.to_csv(ALERT_REPORT_PATH, index=False)
    log.info(f"Alert CSV saved : {ALERT_REPORT_PATH}")

    # ── save human-readable summary ────────────────────────────────────────
    summary_path = os.path.join(ALERTS_DIR, "alert_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  IoT ANOMALY DETECTION — ALERT SUMMARY\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total anomalies detected : {len(alerts):,}\n")
        f.write(f"  CRITICAL               : {n_critical:,}\n")
        f.write(f"  WARNING                : {n_warning:,}\n")
        f.write(f"  INFO                   : {n_info:,}\n")
        f.write(f"  Sensor clusters (bursts): {n_clusters}\n\n")

        f.write(f"Off-hours anomalies      : {off_hours_pct:.1f}% of all alerts\n\n")

        f.write("Most Affected Zones:\n")
        for zone, count in top_zones.items():
            f.write(f"  {zone}: {count} alerts\n")

        f.write("\nMost Flagged Sensors:\n")
        for sid, count in top_sensors.items():
            f.write(f"  {sid}: {count} alerts\n")

        if n_clusters > 0:
            f.write("\nCLUSTER ALERTS (sensor bursts):\n")
            for _, row in clusters.iterrows():
                f.write(f"  {row['sensor_id']} ({row['zone']}) — "
                        f"{row['anomaly_count']} anomalies at {row['hour_bucket']}\n")

        f.write("\n" + "=" * 60 + "\n")

    log.info(f"Alert summary saved : {summary_path}")

    # ── print key numbers ──────────────────────────────────────────────────
    print("\n── Alert Report ──")
    print(f"Total alerts : {len(alerts):,}")
    print(f"CRITICAL     : {n_critical:,}")
    print(f"WARNING      : {n_warning:,}")
    print(f"INFO         : {n_info:,}")
    print(f"Clusters     : {n_clusters}")
    print(f"Off-hours    : {off_hours_pct:.1f}%")
    print(f"Report saved : {ALERT_REPORT_PATH}")

    return alerts_sorted


if __name__ == "__main__":
    df = pd.read_csv("outputs/test_predictions.csv")
    alerts = generate_alert_report(df)
    print(alerts[["sensor_id", "zone", "severity", "if_score"]].head(20))
