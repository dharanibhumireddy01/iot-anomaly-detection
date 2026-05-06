# main.py
# Runs the full IoT anomaly detection pipeline from start to finish.
#
# Usage:
#   python main.py                 run full pipeline on 1M records
#   python main.py --quick         use 200K rows, faster for testing
#   python main.py --skip-gen      skip data generation, use existing CSV
#   python main.py --run-tuning    run hyperparameter grid search
#
# After running you'll find:
#   data/           - raw + feature datasets
#   outputs/models/ - saved model and scaler
#   outputs/alerts/ - alert report CSV and summary text
#   outputs/figures - all 8 visualizations
#   logs/           - full pipeline log

import argparse
import os
import sys
import time

import pandas as pd

# All imports go through src.* so the package structure is clean
from src.data_generator        import generate
from src.feature_engineering   import engineer_features
from src.arima_model           import run_arima, evaluate_arima
from src.isolation_forest_model import run_pipeline
from src.alerting              import generate_alert_report
from src.visualizations        import generate_all
from src.logger                import get_logger
from src.config                import RAW_DATA_PATH, FEATURES_PATH

log = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="IoT Anomaly Detection Pipeline")
    parser.add_argument("--quick",       action="store_true",
                        help="Use 200K rows instead of 1M (faster)")
    parser.add_argument("--skip-gen",    action="store_true",
                        help="Skip data generation, load existing CSV")
    parser.add_argument("--run-tuning",  action="store_true",
                        help="Run hyperparameter grid search (slow)")
    parser.add_argument("--skip-arima",  action="store_true",
                        help="Skip ARIMA step (saves time on large datasets)")
    return parser.parse_args()


def main():
    args    = parse_args()
    t_start = time.time()

    log.info("=" * 55)
    log.info("  IoT Anomaly & Fraud Detection Pipeline")
    log.info("  Author: Dharani Bhumireddy")
    log.info("=" * 55)

    # ── directories ────────────────────────────────────────────────────────
    for d in ["data", "outputs", "outputs/models",
              "outputs/figures", "outputs/alerts", "logs"]:
        os.makedirs(d, exist_ok=True)

    # ── step 1: data ───────────────────────────────────────────────────────
    log.info("STEP 1 — Data")
    if args.skip_gen and os.path.exists(RAW_DATA_PATH):
        nrows = 200_000 if args.quick else None
        df_raw = pd.read_csv(RAW_DATA_PATH, nrows=nrows)
        log.info(f"Loaded {len(df_raw):,} rows from {RAW_DATA_PATH}")
    else:
        df_raw = generate()
        if args.quick:
            df_raw = df_raw.head(200_000).reset_index(drop=True)
            log.info("Quick mode: using 200K rows")

    log.info(f"Anomaly rate: {df_raw['is_anomaly'].mean()*100:.2f}%")

    # ── step 2: features ──────────────────────────────────────────────────
    log.info("STEP 2 — Feature Engineering")
    df_feat = engineer_features(df_raw)
    df_feat.to_csv(FEATURES_PATH, index=False)
    log.info(f"Feature matrix shape: {df_feat.shape}")

    # ── step 3: arima baseline ────────────────────────────────────────────
    df_arima_result = None
    if not args.skip_arima:
        log.info("STEP 3 — ARIMA Baseline (sample of 50K rows)")
        n_arima = min(50_000, len(df_feat))
        df_for_arima = df_feat.head(n_arima).copy()
        df_for_arima = run_arima(df_for_arima)
        evaluate_arima(df_for_arima)
        df_arima_result = df_for_arima
    else:
        log.info("STEP 3 — Skipped (--skip-arima flag)")

    # ── step 4: isolation forest ──────────────────────────────────────────
    log.info("STEP 4 — Isolation Forest")
    df_preds, metrics = run_pipeline(df_feat, run_tuning=args.run_tuning)

    # ── step 5: alerting ──────────────────────────────────────────────────
    log.info("STEP 5 — Alert Report")
    generate_alert_report(df_preds)

    # ── step 6: visualizations ────────────────────────────────────────────
    log.info("STEP 6 — Visualizations")

    # Attach arima columns to preds if available (for the residuals plot)
    if df_arima_result is not None:
        arima_cols = ["sensor_id", "timestamp",
                      "arima_fitted", "arima_residual", "arima_anomaly_flag"]
        df_preds = df_preds.merge(
            df_arima_result[arima_cols], on=["sensor_id", "timestamp"], how="left"
        )

    generate_all(df_raw, df_preds)

    # ── summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info("=" * 55)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Records processed : {len(df_feat):,}")
    log.info(f"  Accuracy          : {metrics.get('accuracy', 0)*100:.2f}%")
    log.info(f"  TPR               : {metrics.get('tpr', 0):.4f}")
    log.info(f"  FPR               : {metrics.get('fpr', 0):.4f}")
    log.info(f"  AUC-ROC           : {metrics.get('auc_roc', 0):.4f}")
    log.info(f"  Time              : {elapsed:.1f}s")
    log.info("=" * 55)

    print("\n  Outputs:")
    print("    data/iot_features.csv")
    print("    outputs/models/isolation_forest.pkl")
    print("    outputs/alerts/alert_report.csv")
    print("    outputs/alerts/alert_summary.txt")
    print("    outputs/figures/  (8 charts)")
    print("    logs/pipeline.log")


if __name__ == "__main__":
    main()
