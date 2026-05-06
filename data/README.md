# Data

The raw dataset is not committed to this repository because it is 1M+ rows (~80MB).

## How to generate it

```bash
# From the project root:
python main.py --quick     # generates 200K rows (fast, ~2 min)
python main.py             # generates full 1M rows (~10 min)
```

This will create:
- `data/iot_sensor_data.csv` — raw sensor readings with injected anomalies
- `data/iot_features.csv`    — engineered feature matrix ready for model training

## What the data looks like

| Column | Description |
|---|---|
| timestamp | Reading datetime (2022–2024) |
| sensor_id | Sensor identifier (SENSOR_001 to SENSOR_050) |
| zone | Zone grouping (ZONE_A, B, C, D) |
| temperature | Temperature reading in °F |
| pressure | Pressure reading in PSI |
| vibration | Vibration level |
| power_draw | Power consumption |
| is_anomaly | Ground truth label (0 = normal, 1 = anomaly) |
| anomaly_type | Type: normal / spike / flatline / drift / pattern_break |
