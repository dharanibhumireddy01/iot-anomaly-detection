# config.py
# All project-wide constants live here.
# If you need to change anything — data size, model params, thresholds —
# change it here and it flows through the whole pipeline automatically.

import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
MODEL_DIR     = os.path.join(OUTPUT_DIR, "models")
FIGURES_DIR   = os.path.join(OUTPUT_DIR, "figures")
ALERTS_DIR    = os.path.join(OUTPUT_DIR, "alerts")
LOGS_DIR      = os.path.join(BASE_DIR,   "logs")

RAW_DATA_PATH        = os.path.join(DATA_DIR, "iot_sensor_data.csv")
FEATURES_PATH        = os.path.join(DATA_DIR, "iot_features.csv")
PREDICTIONS_PATH     = os.path.join(OUTPUT_DIR, "test_predictions.csv")
MODEL_PATH           = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH          = os.path.join(MODEL_DIR, "scaler.pkl")
ALERT_REPORT_PATH    = os.path.join(ALERTS_DIR, "alert_report.csv")
LOG_FILE             = os.path.join(LOGS_DIR, "pipeline.log")

# ── data generation ────────────────────────────────────────────────────────
RANDOM_SEED      = 42
N_SENSORS        = 50
N_RECORDS        = 1_000_000
START_DATE       = "2022-01-01"
END_DATE         = "2024-01-01"
ANOMALY_RATE     = 0.03          # 3% of records will be anomalies

SENSOR_IDS = [f"SENSOR_{str(i).zfill(3)}" for i in range(1, N_SENSORS + 1)]

# Zone groupings
SENSOR_ZONES = {
    "ZONE_A": SENSOR_IDS[:15],    # industrial floor
    "ZONE_B": SENSOR_IDS[15:30],  # HVAC units
    "ZONE_C": SENSOR_IDS[30:40],  # power distribution
    "ZONE_D": SENSOR_IDS[40:],    # outdoor perimeter
}

# Normal operating ranges per zone
# (temp_mu, temp_sd, press_mu, press_sd, vib_mu, vib_sd, power_mu, power_sd)
ZONE_PARAMS = {
    "ZONE_A": dict(temp_mu=72,  temp_sd=4,  press_mu=14.5, press_sd=0.5,
                   vib_mu=0.12, vib_sd=0.03, power_mu=220, power_sd=10),
    "ZONE_B": dict(temp_mu=65,  temp_sd=3,  press_mu=13.0, press_sd=0.4,
                   vib_mu=0.08, vib_sd=0.02, power_mu=110, power_sd=8),
    "ZONE_C": dict(temp_mu=80,  temp_sd=5,  press_mu=15.0, press_sd=0.6,
                   vib_mu=0.15, vib_sd=0.04, power_mu=440, power_sd=20),
    "ZONE_D": dict(temp_mu=55,  temp_sd=8,  press_mu=14.2, press_sd=0.3,
                   vib_mu=0.05, vib_sd=0.02, power_mu=24,  power_sd=3),
}

# Anomaly injection weights (must sum to 1.0)
ANOMALY_TYPE_WEIGHTS = {
    "spike":         0.35,
    "flatline":      0.25,
    "drift":         0.25,
    "pattern_break": 0.15,
}

# ── feature engineering ────────────────────────────────────────────────────
SENSOR_COLS  = ["temperature", "pressure", "vibration", "power_draw"]
ROLL_WINDOWS = [5, 15, 30]
LAG_STEPS    = [1, 2, 3]

# ── ARIMA ──────────────────────────────────────────────────────────────────
ARIMA_ORDER       = (2, 1, 2)
ARIMA_TARGET_COL  = "temperature"
ARIMA_RESID_SIGMA = 3.0          # flag residuals beyond 3 std deviations
ARIMA_MIN_ROWS    = 50           # skip sensors with fewer records than this

# ── Isolation Forest ───────────────────────────────────────────────────────
# These are the best params found after grid search
IF_BEST_PARAMS = {
    "n_estimators":  300,
    "max_samples":   0.8,
    "contamination": 0.03,
    "max_features":  1.0,
    "random_state":  RANDOM_SEED,
    "n_jobs":        -1,
}

# Grid search space (used when --run-tuning flag is passed)
IF_PARAM_GRID = {
    "n_estimators":  [100, 200, 300],
    "max_samples":   [0.6, 0.8, "auto"],
    "contamination": [0.02, 0.03, 0.05],
    "max_features":  [0.7, 1.0],
    "random_state":  [RANDOM_SEED],
}

TRAIN_SPLIT = 0.80   # 80% train, 20% test (time-based split)

# ── alerting ───────────────────────────────────────────────────────────────
# Anomaly score threshold for a CRITICAL alert (lower = more anomalous)
CRITICAL_SCORE_THRESHOLD = -0.15
# Minimum anomaly score for a WARNING alert
WARNING_SCORE_THRESHOLD  = -0.10
# How many alerts before we call it a "cluster" (suspicious burst)
CLUSTER_THRESHOLD        = 5

# ── visualizations ─────────────────────────────────────────────────────────
PLOT_DPI    = 150
COLOR_NORMAL  = "#2E86AB"
COLOR_ANOMALY = "#E84855"
COLOR_ACCENT  = "#F7B731"
COLOR_FITTED  = "#7BC67E"
FIG_SIZE_WIDE = (15, 5)
FIG_SIZE_STD  = (12, 5)
FIG_SIZE_SQ   = (7, 6)
