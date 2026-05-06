# visualizations.py
# All charts for the project.
# Every function saves a PNG to outputs/figures/ and returns the figure.
# Call generate_all() to run everything at once.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.config import (
    FIGURES_DIR, PLOT_DPI,
    COLOR_NORMAL, COLOR_ANOMALY, COLOR_ACCENT, COLOR_FITTED
)
from src.logger import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_anomaly_distribution(df: pd.DataFrame) -> None:
    """Anomaly type breakdown and zone distribution side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Anomaly Distribution — IoT Sensor Network", fontsize=13, fontweight="bold")

    # left panel: counts by type
    type_vc = df["anomaly_type"].value_counts()
    colors  = [COLOR_ANOMALY if t != "normal" else COLOR_NORMAL for t in type_vc.index]
    bars    = axes[0].bar(type_vc.index, type_vc.values, color=colors, edgecolor="white", width=0.6)
    axes[0].set_title("Records by Anomaly Type", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 200,
                     f"{int(bar.get_height()):,}",
                     ha="center", va="bottom", fontsize=9)

    # right panel: anomalies per zone
    zone_c = df[df["is_anomaly"] == 1].groupby("zone").size()
    axes[1].bar(zone_c.index, zone_c.values, color=COLOR_ACCENT, edgecolor="white", width=0.5)
    axes[1].set_title("Anomaly Count per Zone", fontweight="bold")
    axes[1].set_ylabel("Anomalies")

    plt.tight_layout()
    _save(fig, "01_anomaly_distribution.png")


def plot_sensor_timeseries(df: pd.DataFrame,
                            sensor_id: str = "SENSOR_001",
                            col: str = "temperature") -> None:
    """
    Time-series line chart for one sensor showing all readings with
    anomalies highlighted as red scatter points.
    """
    sdf = df[df["sensor_id"] == sensor_id].copy()
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    sdf = sdf.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.plot(sdf["timestamp"], sdf[col],
            color=COLOR_NORMAL, linewidth=0.5, alpha=0.8, label="Normal reading")

    anom = sdf[sdf["is_anomaly"] == 1]
    ax.scatter(anom["timestamp"], anom[col],
               color=COLOR_ANOMALY, s=15, zorder=5, label="Anomaly", alpha=0.9)

    ax.set_title(f"{col.replace('_',' ').title()} — {sensor_id}  (anomalies in red)",
                 fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(col.replace("_", " ").title())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=25)
    ax.legend()
    plt.tight_layout()
    _save(fig, "02_sensor_timeseries.png")


def plot_rolling_stats(df: pd.DataFrame,
                        sensor_id: str = "SENSOR_001") -> None:
    """
    Shows the raw temperature reading, rolling mean (15-window),
    and rolling std dev for one sensor with anomaly overlay.
    """
    needed_cols = ["temperature_roll_mean_15", "temperature_roll_std_15"]
    for c in needed_cols:
        if c not in df.columns:
            log.warning("Rolling features not found — skipping rolling stats plot")
            return

    sdf = df[df["sensor_id"] == sensor_id].copy()
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    sdf = sdf.sort_values("timestamp")
    anom = sdf[sdf["is_anomaly"] == 1]

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(f"Rolling Statistics — {sensor_id}", fontweight="bold")

    axes[0].plot(sdf["timestamp"], sdf["temperature"],
                 color="#AACBDE", linewidth=0.4, alpha=0.7, label="Raw")
    axes[0].plot(sdf["timestamp"], sdf["temperature_roll_mean_15"],
                 color=COLOR_NORMAL, linewidth=1.2, label="Rolling Mean (15)")
    axes[0].scatter(anom["timestamp"], anom["temperature"],
                    color=COLOR_ANOMALY, s=10, zorder=5, label="Anomaly")
    axes[0].set_ylabel("Temperature (°F)")
    axes[0].legend(fontsize=9)

    axes[1].plot(sdf["timestamp"], sdf["temperature_roll_std_15"],
                 color=COLOR_ACCENT, linewidth=0.8, label="Rolling Std (15)")
    axes[1].set_ylabel("Std Dev")
    axes[1].set_xlabel("Date")
    axes[1].legend(fontsize=9)

    plt.xticks(rotation=25)
    plt.tight_layout()
    _save(fig, "03_rolling_statistics.png")


def plot_zone_hour_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap showing anomaly rate (%) by zone and hour of day.
    High values at 2am in ZONE_C would be a red flag in production.
    """
    if "hour_raw" not in df.columns:
        df["hour_raw"] = pd.to_datetime(df["timestamp"]).dt.hour

    pivot = (
        df.groupby(["zone", "hour_raw"])["is_anomaly"]
        .mean()
        .mul(100)
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, cmap="YlOrRd", annot=False,
                linewidths=0.3, linecolor="#eeeeee", ax=ax)
    ax.set_title("Anomaly Rate (%) by Zone × Hour of Day", fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Zone")
    plt.tight_layout()
    _save(fig, "04_zone_hour_heatmap.png")


def plot_anomaly_score_dist(df_preds: pd.DataFrame) -> None:
    """
    Distribution of Isolation Forest anomaly scores split by true label.
    Shows how well the model separates normal from anomalous readings.
    """
    if "if_score" not in df_preds.columns:
        log.warning("if_score column not found — skipping score distribution plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, color, name in [
        (0, COLOR_NORMAL,  "Normal"),
        (1, COLOR_ANOMALY, "Anomaly"),
    ]:
        subset = df_preds[df_preds["is_anomaly"] == label]["if_score"].dropna()
        ax.hist(subset, bins=80, alpha=0.65, color=color, label=name, density=True)

    ax.set_title("Isolation Forest — Anomaly Score Distribution", fontweight="bold")
    ax.set_xlabel("Anomaly Score  (more negative = more anomalous)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    _save(fig, "05_anomaly_score_distribution.png")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Styled confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Pred Normal", "Pred Anomaly"],
                yticklabels=["True Normal", "True Anomaly"],
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14}, ax=ax)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.set_title(f"Confusion Matrix  |  Accuracy: {acc*100:.2f}%", fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, "06_confusion_matrix.png")


def plot_precision_recall(y_true: np.ndarray, scores: np.ndarray) -> None:
    """Precision-Recall curve with Average Precision score."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, -scores)
    ap = average_precision_score(y_true, -scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rec, prec, color=COLOR_ANOMALY, linewidth=2, label=f"AP = {ap:.3f}")
    ax.fill_between(rec, prec, alpha=0.1, color=COLOR_ANOMALY)
    ax.axhline(y=y_true.mean(), color="gray", linestyle="--",
               linewidth=1, label=f"Random baseline ({y_true.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Isolation Forest", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, "07_precision_recall_curve.png")


def plot_arima_residuals(df: pd.DataFrame,
                          sensor_id: str = "SENSOR_001") -> None:
    """
    Shows ARIMA predicted vs actual temperature for one sensor,
    plus the residuals. Spikes in the residual line correspond to anomalies.
    """
    if "arima_fitted" not in df.columns:
        log.warning("ARIMA columns not found — skipping ARIMA residuals plot")
        return

    sdf = df[df["sensor_id"] == sensor_id].copy()
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    sdf = sdf.sort_values("timestamp").head(500)  # zoom in for clarity

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    fig.suptitle(f"ARIMA Baseline — {sensor_id}", fontweight="bold")

    axes[0].plot(sdf["timestamp"], sdf["temperature"],
                 color="#AACBDE", linewidth=0.6, label="Actual")
    axes[0].plot(sdf["timestamp"], sdf["arima_fitted"],
                 color=COLOR_FITTED, linewidth=1.2, alpha=0.8, label="ARIMA Predicted")
    axes[0].set_ylabel("Temperature (°F)")
    axes[0].legend()

    axes[1].plot(sdf["timestamp"], sdf["arima_residual"],
                 color=COLOR_ACCENT, linewidth=0.6)
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Residual (actual − predicted)")
    axes[1].set_xlabel("Timestamp")
    axes[1].set_title("Residuals — spikes indicate potential anomalies")

    plt.tight_layout()
    _save(fig, "08_arima_residuals.png")


def generate_all(df_raw: pd.DataFrame,
                  df_preds: pd.DataFrame = None) -> None:
    """Run all visualizations in one call."""
    log.info("Generating all visualizations...")

    plot_anomaly_distribution(df_raw)
    plot_sensor_timeseries(df_raw)
    plot_rolling_stats(df_raw)
    plot_zone_hour_heatmap(df_raw)

    if df_preds is not None:
        plot_anomaly_score_dist(df_preds)
        plot_confusion_matrix(df_preds["is_anomaly"].values, df_preds["if_pred"].values)
        plot_precision_recall(df_preds["is_anomaly"].values, df_preds["if_score"].values)

        if "arima_fitted" in df_preds.columns:
            plot_arima_residuals(df_preds)

    log.info(f"All figures saved to: {FIGURES_DIR}")
