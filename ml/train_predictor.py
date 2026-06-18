"""
환율 예측 모델 학습 스크립트
- 3개 모델 (Linear, Ridge, RandomForest) 학습 및 비교
- MLflow에 실험 기록
- 최고 성능 모델을 models/predictor/current.pkl로 저장
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "predictor"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# MLflow 설정 (로컬 파일 기반)
MLFLOW_DB = ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT_DIR / "mlruns"
MLFLOW_ARTIFACTS.mkdir(exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("rate_prediction")


def load_data_from_db():
    """DB에서 KRW 환율 데이터 로드. 부족하면 더미 데이터 생성."""
    db_path = ROOT_DIR / "rate_dashboard.db"
    if db_path.exists():
        engine = create_engine(f"sqlite:///{db_path}")
        df = pd.read_sql(
            "SELECT rate, collected_at FROM exchange_rates "
            "WHERE target_currency='KRW' ORDER BY collected_at ASC",
            engine,
        )
        if len(df) >= 30:
            print(f"[INFO] DB에서 {len(df)}건 로드")
            return df["rate"].astype(float).tolist()

    # 데이터 부족 시 더미 데이터 생성 (학습용 합성 데이터)
    print("[INFO] DB 데이터 부족 → 더미 환율 시계열 생성 (100건)")
    np.random.seed(42)
    base = 1350.0
    rates = [base]
    for _ in range(99):
        change = float(np.random.normal(0, 5))
        next_rate = rates[-1] + change
        next_rate = max(1200.0, min(1500.0, next_rate))
        rates.append(next_rate)
    print(f"[INFO] 생성된 더미 데이터 길이: {len(rates)}")
    return rates


def make_features(rates: np.ndarray):
    """
    피처 엔지니어링:
    - lag_1, lag_3, lag_7: 과거 1/3/7일 환율
    - rolling_mean_7: 7일 이동평균
    - rolling_std_7: 7일 이동 표준편차
    타겟: 다음날 환율
    """
    rates = np.asarray(rates, dtype=float).flatten()
    df = pd.DataFrame({"rate": rates})
    df["lag_1"] = df["rate"].shift(1)
    df["lag_3"] = df["rate"].shift(3)
    df["lag_7"] = df["rate"].shift(7)
    df["rolling_mean_7"] = df["rate"].rolling(7).mean()
    df["rolling_std_7"] = df["rate"].rolling(7).std()
    df["target"] = df["rate"].shift(-1)  # 다음날 환율
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["lag_1", "lag_3", "lag_7", "rolling_mean_7", "rolling_std_7"]
    X = df[feature_cols].values
    y = df["target"].values
    return X, y, feature_cols


def train_one_model(name, model, X_train, X_test, y_train, y_test, feature_cols):
    """단일 모델 학습 + MLflow 기록"""
    with mlflow.start_run(run_name=name) as run:
        # 학습
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 평가 지표
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        # MLflow 기록: parameter
        mlflow.log_param("model_name", name)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("features", ",".join(feature_cols))
        if hasattr(model, "get_params"):
            for k, v in model.get_params().items():
                mlflow.log_param(f"hp_{k}", v)

        # MLflow 기록: metric
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # MLflow 기록: model (artifact)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[{name}] RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        return {
            "name": name,
            "model": model,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "run_id": run.info.run_id,
        }


def save_best_model(best_result, feature_cols):
    """최고 성능 모델을 운영 디렉토리에 저장 + 메타데이터 기록"""
    metadata_path = MODEL_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            old_meta = json.load(f)
        next_version = old_meta.get("version_number", 0) + 1
    else:
        next_version = 1

    version_str = f"v{next_version}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = MODEL_DIR / f"{version_str}_{timestamp}.pkl"
    current_path = MODEL_DIR / "current.pkl"

    # 모델 파일 2개 저장: 버전 백업 + current
    joblib.dump(best_result["model"], versioned_path)
    joblib.dump(best_result["model"], current_path)

    # 메타데이터 기록
    metadata = {
        "version": version_str,
        "version_number": next_version,
        "model_name": best_result["name"],
        "trained_at": timestamp,
        "metrics": {
            "rmse": best_result["rmse"],
            "mae": best_result["mae"],
            "r2": best_result["r2"],
        },
        "feature_cols": feature_cols,
        "mlflow_run_id": best_result["run_id"],
        "versioned_file": versioned_path.name,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVE] {version_str} ({best_result['name']}) → {current_path}")
    print(f"[SAVE] 백업: {versioned_path.name}")
    print(f"[SAVE] 메타데이터: {metadata_path}")


def main():
    print("=" * 60)
    print("환율 예측 모델 학습 시작")
    print("=" * 60)

    # 1. 데이터 로드
    rates = load_data_from_db()
    X, y, feature_cols = make_features(rates)
    print(f"[INFO] 전체 샘플 수: {len(X)}, 피처 수: {len(feature_cols)}")

    # 2. Train/Test 분할 (시계열이므로 시간 순서 유지, 마지막 20%를 테스트)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

    # 3. 3개 모델 학습 & MLflow 기록
    models = [
        ("linear_regression", LinearRegression()),
        ("ridge", Ridge(alpha=1.0)),
        ("random_forest", RandomForestRegressor(n_estimators=50, random_state=42)),
    ]
    results = []
    for name, model in models:
        results.append(
            train_one_model(name, model, X_train, X_test, y_train, y_test, feature_cols)
        )

    # 4. 최고 성능 모델 선정 (RMSE 최소)
    best = min(results, key=lambda r: r["rmse"])
    print("\n" + "=" * 60)
    print(f"[BEST] {best['name']} (RMSE={best['rmse']:.4f})")
    print("=" * 60)

    # 5. 운영 모델 저장
    save_best_model(best, feature_cols)


if __name__ == "__main__":
    main()