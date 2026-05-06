# isolation_forest_model.py
# The main ML model for anomaly detection.
#
# Isolation Forest works by building random decision trees and measuring
# how many splits it takes to isolate each data point. Normal points
# need many splits (they're dense, similar to everything else).
# Anomalies need very few splits (they're rare and different).
#
# Why Isolation Forest and not something like One-Class SVM or LOF?
#   - Scales to 1M+ records (LOF doesn't)
#   - No assumption of normality (unlike OCSVM)
#   - Works without labeled data (unsupervised)
#   - Fast training and inference
#   - Interpretable contamination parameter

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score
)

from src.config import (
    IF_BEST_PARAMS, IF_PARAM_GRID, TRAIN_SPLIT,
    MODEL_PATH, SCALER_PATH, PREDICTIONS_PATH
)
from src.logger import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore")


def _meta_cols() -> set:
    """Columns that are not features — should never be passed to the model."""
    return {
        "timestamp", "sensor_id", "zone",
        "is_anomaly", "anomaly_type",
        "arima_residual", "arima_fitted", "arima_anomaly_flag",
        "hour_raw",
    }


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _meta_cols()]


def _scale(df: pd.DataFrame,
           scaler: StandardScaler = None,
           fit: bool = True) -> Tuple[np.ndarray, StandardScaler]:
    """
    Extract feature matrix and apply StandardScaler.
    If fit=True, a new scaler is fitted. Otherwise use the passed scaler.
    """
    X = df[get_feature_cols(df)].values.astype(np.float32)

    if fit:
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    else:
        return scaler.transform(X), scaler


def tune(df_train: pd.DataFrame, df_val: pd.DataFrame) -> Dict:
    """
    Grid search over IF_PARAM_GRID.
    Picks the combination that maximises F1 score on the validation set.
    Returns the best parameter dictionary.
    """
    X_train, scaler = _scale(df_train, fit=True)
    X_val,   _      = _scale(df_val, scaler=scaler, fit=False)
    y_val           = df_val["is_anomaly"].values

    best_f1     = -1
    best_params = None
    grid        = list(ParameterGrid(IF_PARAM_GRID))

    log.info(f"Starting hyperparameter search over {len(grid)} combinations")

    for i, params in enumerate(grid, 1):
        clf    = IsolationForest(**params, n_jobs=-1)
        clf.fit(X_train)

        raw    = clf.predict(X_val)
        y_pred = np.where(raw == -1, 1, 0)
        f1     = f1_score(y_val, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1     = f1
            best_params = params

        if i % 5 == 0:
            log.info(f"  {i}/{len(grid)} done | best F1 so far: {best_f1:.4f}")

    log.info(f"Best params: {best_params}")
    log.info(f"Best F1    : {best_f1:.4f}")
    return best_params


def train(df_train: pd.DataFrame,
          params: Dict = None) -> Tuple[IsolationForest, StandardScaler]:
    """
    Train the Isolation Forest on the training set.
    Returns the fitted model and the fitted scaler.
    """
    if params is None:
        params = IF_BEST_PARAMS

    log.info(f"Training Isolation Forest | rows: {len(df_train):,} | params: {params}")

    X_train, scaler = _scale(df_train, fit=True)
    clf = IsolationForest(**params)
    clf.fit(X_train)

    log.info("Training complete")
    return clf, scaler


def predict(clf: IsolationForest,
            scaler: StandardScaler,
            df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run predictions on a DataFrame.
    Returns:
        y_pred  — binary array: 1 = anomaly, 0 = normal
        scores  — raw anomaly scores (more negative = more anomalous)
    """
    X, _ = _scale(df, scaler=scaler, fit=False)
    raw    = clf.predict(X)
    scores = clf.score_samples(X)
    y_pred = np.where(raw == -1, 1, 0)
    return y_pred, scores


def evaluate(y_true: np.ndarray,
             y_pred: np.ndarray,
             scores: np.ndarray = None) -> Dict:
    """
    Full evaluation.
    Prints the classification report, confusion matrix,
    false positive rate, and AUC-ROC if scores are available.
    Returns a dict of key metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n── Isolation Forest Results ──")
    print(f"Accuracy           : {acc*100:.2f}%")
    print(f"True Positive Rate : {tpr:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    print(f"Confusion Matrix:\n{cm}")

    metrics = dict(accuracy=acc, tpr=tpr, fpr=fpr,
                   tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))

    if scores is not None:
        auc = roc_auc_score(y_true, -scores)
        print(f"\nAUC-ROC: {auc:.4f}")
        metrics["auc_roc"] = auc

    return metrics


def save_model(clf: IsolationForest, scaler: StandardScaler) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"Model saved  : {MODEL_PATH}")
    log.info(f"Scaler saved : {SCALER_PATH}")


def load_model() -> Tuple[IsolationForest, StandardScaler]:
    clf    = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    log.info("Model and scaler loaded")
    return clf, scaler


def run_pipeline(df: pd.DataFrame,
                 run_tuning: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Full train-test pipeline:
        1. Time-based split (no leakage — train on past, test on future)
        2. Optional hyperparameter tuning
        3. Train Isolation Forest
        4. Predict and evaluate on test set
        5. Save model and predictions
    """
    # ── split ──────────────────────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * TRAIN_SPLIT)
    df_train = df.iloc[:cut].copy()
    df_test  = df.iloc[cut:].copy()

    log.info(f"Train: {len(df_train):,} | Test: {len(df_test):,}")
    log.info(f"Anomaly rate — train: {df_train['is_anomaly'].mean()*100:.2f}%  "
             f"test: {df_test['is_anomaly'].mean()*100:.2f}%")

    # ── tune or use best known params ──────────────────────────────────────
    if run_tuning:
        val_cut  = int(len(df_train) * 0.85)
        best_p   = tune(df_train.iloc[:val_cut], df_train.iloc[val_cut:])
    else:
        best_p = IF_BEST_PARAMS
        log.info(f"Using pre-tuned params: {best_p}")

    # ── train ──────────────────────────────────────────────────────────────
    clf, scaler = train(df_train, params=best_p)

    # ── predict ────────────────────────────────────────────────────────────
    y_pred, scores = predict(clf, scaler, df_test)
    y_true = df_test["is_anomaly"].values

    # ── evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate(y_true, y_pred, scores)

    # ── save outputs ───────────────────────────────────────────────────────
    df_test = df_test.copy()
    df_test["if_pred"]  = y_pred
    df_test["if_score"] = scores

    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    df_test.to_csv(PREDICTIONS_PATH, index=False)
    log.info(f"Predictions saved: {PREDICTIONS_PATH}")

    save_model(clf, scaler)

    return df_test, metrics


if __name__ == "__main__":
    from src.feature_engineering import engineer_features

    df_raw  = pd.read_csv("data/iot_sensor_data.csv", nrows=200_000)
    df_feat = engineer_features(df_raw)
    df_out, metrics = run_pipeline(df_feat)

    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {round(v,4) if isinstance(v, float) else v}")
