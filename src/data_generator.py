# data_generator.py
# Builds a synthetic IoT sensor dataset that looks and behaves like
# real industrial sensor data — complete with normal operating patterns
# and four different types of injected anomalies.
#
# Why synthetic? Because real IoT fraud datasets are proprietary.
# The patterns here mirror real-world abuse signatures documented in
# industrial IoT security literature.

import os
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import (
    RANDOM_SEED, N_SENSORS, N_RECORDS, START_DATE, END_DATE,
    ANOMALY_RATE, SENSOR_IDS, SENSOR_ZONES, ZONE_PARAMS,
    ANOMALY_TYPE_WEIGHTS, RAW_DATA_PATH
)
from src.logger import get_logger

log = get_logger(__name__)

np.random.seed(RANDOM_SEED)

# Build a reverse map: sensor_id -> zone name
ZONE_MAP = {
    sid: zone
    for zone, sids in SENSOR_ZONES.items()
    for sid in sids
}


def _business_hours_multiplier(hour: int) -> float:
    """
    Sensors in industrial settings run harder during business hours.
    This gives readings a realistic daily rhythm using a sine curve
    that peaks around 1pm and dips overnight.
    """
    if 8 <= hour <= 18:
        return 1.0 + 0.15 * np.sin(np.pi * (hour - 8) / 10)
    return 0.85


def _make_timestamps(n: int) -> np.ndarray:
    """
    Generate n random timestamps spread across the date range.
    Sorted so each sensor's data is chronological.
    """
    start_ts = int(pd.Timestamp(START_DATE).timestamp())
    end_ts   = int(pd.Timestamp(END_DATE).timestamp())
    offsets  = np.sort(np.random.randint(start_ts, end_ts, size=n))
    return pd.to_datetime(offsets, unit="s")


def _normal_readings(sensor_id: str, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate clean, normally-distributed sensor readings for one sensor.
    Applies a time-of-day multiplier so readings are higher during
    business hours — matches industrial equipment behaviour.
    """
    zone   = ZONE_MAP[sensor_id]
    p      = ZONE_PARAMS[zone]
    n      = len(timestamps)
    tod    = np.array([_business_hours_multiplier(h) for h in timestamps.hour])

    temperature = np.random.normal(p["temp_mu"],  p["temp_sd"],  n) * tod
    pressure    = np.random.normal(p["press_mu"], p["press_sd"], n)
    vibration   = np.abs(np.random.normal(p["vib_mu"], p["vib_sd"], n)) * tod
    power_draw  = np.random.normal(p["power_mu"], p["power_sd"], n) * tod

    return pd.DataFrame({
        "timestamp":    timestamps,
        "sensor_id":    sensor_id,
        "zone":         zone,
        "temperature":  np.round(temperature, 2),
        "pressure":     np.round(pressure, 3),
        "vibration":    np.round(np.abs(vibration), 4),
        "power_draw":   np.round(power_draw, 2),
        "is_anomaly":   0,
        "anomaly_type": "normal",
    })


def _inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly inject four types of anomalies that map to real-world
    fraud and abuse patterns:

    spike         - sudden extreme reading (tampering, power surge, meter fraud)
    flatline      - value stuck constant (dead sensor, spoofed readings)
    drift         - slow creep upward (calibration drift, refrigerant leak)
    pattern_break - unusual activity at 2-4am (unauthorized after-hours access)

    Each type has a realistic weight based on how common it is in
    real industrial incident data.
    """
    df = df.copy()

    n_inject   = int(len(df) * ANOMALY_RATE)
    anom_types = list(ANOMALY_TYPE_WEIGHTS.keys())
    weights    = list(ANOMALY_TYPE_WEIGHTS.values())

    chosen_idx   = np.random.choice(df.index, size=n_inject, replace=False)
    chosen_types = np.random.choice(anom_types, size=n_inject, p=weights)

    for idx, atype in zip(chosen_idx, chosen_types):

        if atype == "spike":
            # Reading jumps to 3.5–6× normal — hard to miss on a dashboard
            multiplier = np.random.uniform(3.5, 6.0)
            df.at[idx, "temperature"] *= multiplier
            df.at[idx, "vibration"]   *= multiplier
            df.at[idx, "power_draw"]  *= np.random.uniform(2.0, 4.0)

        elif atype == "flatline":
            # Readings freeze — sensor is dead or someone is replaying old data
            flat_temp  = df.at[idx, "temperature"]
            flat_press = df.at[idx, "pressure"]
            window     = min(10, len(df) - idx)
            df.loc[idx : idx + window, "temperature"] = flat_temp
            df.loc[idx : idx + window, "pressure"]    = flat_press
            df.loc[idx : idx + window, "vibration"]   = 0.0

        elif atype == "drift":
            # Slow upward creep — easy to miss if you only check latest values
            window = min(20, len(df) - idx)
            drift  = np.linspace(1.0, 2.5, window)
            df.loc[idx : idx + window - 1, "temperature"] *= drift
            df.loc[idx : idx + window - 1, "pressure"]    *= (drift * 0.6)

        elif atype == "pattern_break":
            # Shift the timestamp into the middle of the night
            ts = df.at[idx, "timestamp"]
            df.at[idx, "timestamp"] = ts.replace(
                hour=int(np.random.randint(2, 4)),
                minute=int(np.random.randint(0, 59))
            )
            df.at[idx, "power_draw"] *= np.random.uniform(1.8, 3.0)

        df.at[idx, "is_anomaly"]   = 1
        df.at[idx, "anomaly_type"] = atype

    return df.sort_values("timestamp").reset_index(drop=True)


def generate(output_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Generates the full dataset and saves it to CSV.
    Returns the DataFrame so callers can use it directly.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records_per_sensor = N_RECORDS // N_SENSORS
    log.info(f"Generating {N_RECORDS:,} records across {N_SENSORS} sensors...")

    chunks = []
    for i, sensor_id in enumerate(SENSOR_IDS, 1):
        timestamps = _make_timestamps(records_per_sensor)
        df_sensor  = _normal_readings(sensor_id, timestamps)
        df_sensor  = _inject_anomalies(df_sensor)
        chunks.append(df_sensor)

        if i % 10 == 0:
            log.info(f"  Built {i}/{N_SENSORS} sensors")

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_csv(output_path, index=False)

    n_anom = df["is_anomaly"].sum()
    anom_breakdown = df[df["is_anomaly"] == 1]["anomaly_type"].value_counts().to_dict()

    log.info(f"Dataset saved  : {output_path}")
    log.info(f"Total records  : {len(df):,}")
    log.info(f"Total anomalies: {n_anom:,}  ({n_anom / len(df) * 100:.2f}%)")
    log.info(f"Anomaly types  : {anom_breakdown}")

    return df


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(df["anomaly_type"].value_counts())
