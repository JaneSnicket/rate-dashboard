"""
환율 이상치 탐지 모델 학습 스크립트
- IsolationForest 모델 학습
- MLflow에 실험 기록
- models/detector/current.pkl로 저장
"""
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

# 프로젝트 루트 경로
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "detector"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# MLflow 설정 (SQLite backend)
MLFLOW_DB = ROOT_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("rate_anomaly_detection")


def load_data_from_db():
    """DB에서 KRW 환율 데이터 로드. 부족하면 더미 데이터 생성."""
    db_path = ROOT_DIR / "rate_dashboard.db"
    if db_path.exists():
        engine = create_engine(f"sqlite:///{db_path}")
        df = pd.read_sql(
            "SELECT rate, change_percent FROM exchange_rates "
            "WHERE target_currency='KRW' ORDER BY collected_at ASC",
            engine,
        )
        if len(df) >= 30:
            print(f"[INFO] DB에서 {len(df)}건 로드")
            return df["rate"].astype(float).tolist(), df["change_percent"].astype(float).tolist()

    # 더미 데이터 생성 (정상 + 일부 이상치 섞기)
    print("[INFO] DB 데이터 부족 → 더미 환율 시계열 생성 (정상 95건 + 이상치 5건)")
    np.random.seed(42)
    base = 1350.0
    rates = [base]
    change_percents = [0.0]

    # 정상 데이터 95건
    for _ in range(94):
        change = float(np.random.normal(0, 5))  # 작은 변동
        next_rate = rates[-1] + change
        next_rate = max(1200.0, min(1500.0, next_rate))
        change_pct = (next_rate - rates[-1]) / rates[-1] * 100
        rates.append(next_rate)
        change_percents.append(round(change_pct, 4))

    # 이상치 5건 (급등락) - 인위적으로 큰 변동 추가
    anomaly_changes = [40.0, -35.0, 50.0, -45.0, 38.0]
    for ac in anomaly_changes:
        next_rate = rates[-1] + ac
        next_rate = max(1100.0, min(1600.0, next_rate))
        change_pct = (next_rate - rates[-1]) / rates[-1] * 100
        rates.append(next_rate)
        change_percents.append(round(change_pct, 4))

    print(f"[INFO] 생성된 더미 데이터 길이: {len(rates)}")
    return rates, change_percents


def make_features(rates, change_percents):
    """
    이상치 탐지용 피처:
    - rate: 현재 환율
    - change_percent: 전일 대비 등락률
    - abs_change: 절댓값 등락률 (방향 무관 크기)
    """
    rates = np.asarray(rates, dtype=float).flatten()
    change_percents = np.asarray(change_percents, dtype=float).flatten()
    abs_changes = np.abs(change_percents)

    df = pd.DataFrame({
        "rate": rates,
        "change_percent": change_percents,
        "abs_change": abs_changes,
    })
    feature_cols = ["rate", "change_percent", "abs_change"]
    X = df[feature_cols].values
    return X, feature_cols


def train_detector(X, feature_cols, contamination=0.05):
    """IsolationForest 학습 + MLflow 기록"""
    with mlflow.start_run(run_name="isolation_forest") as run:
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        model.fit(X)

        # 학습 데이터에 대한 예측 (평가용)
        predictions = model.predict(X)  
        scores = model.score_samples(X)  

        n_total = len(X)
        n_anomalies = int(np.sum(predictions == -1))
        anomaly_ratio = n_anomalies / n_total

        # MLflow 기록: parameter
        mlflow.log_param("model_name", "isolation_forest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_samples", n_total)
        mlflow.log_param("features", ",".join(feature_cols))

        # MLflow 기록: metric
        mlflow.log_metric("n_anomalies_detected", n_anomalies)
        mlflow.log_metric("anomaly_ratio", anomaly_ratio)
        mlflow.log_metric("min_score", float(scores.min()))
        mlflow.log_metric("max_score", float(scores.max()))
        mlflow.log_metric("mean_score", float(scores.mean()))

        # MLflow 기록: model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[isolation_forest] 전체 {n_total}건 중 이상치 {n_anomalies}건 탐지 ({anomaly_ratio*100:.2f}%)")
        print(f"[isolation_forest] score 범위: [{scores.min():.4f}, {scores.max():.4f}]")

        return {
            "model": model,
            "n_anomalies": n_anomalies,
            "anomaly_ratio": anomaly_ratio,
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "mean_score": float(scores.mean()),
            "run_id": run.info.run_id,
            "contamination": contamination,
        }


def save_model(result, feature_cols):
    """학습된 모델을 운영 디렉토리에 저장 + 메타데이터 기록"""
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

    joblib.dump(result["model"], versioned_path)
    joblib.dump(result["model"], current_path)

    metadata = {
        "version": version_str,
        "version_number": next_version,
        "model_name": "isolation_forest",
        "trained_at": timestamp,
        "metrics": {
            "n_anomalies_detected": result["n_anomalies"],
            "anomaly_ratio": result["anomaly_ratio"],
            "min_score": result["min_score"],
            "max_score": result["max_score"],
            "mean_score": result["mean_score"],
        },
        "hyperparameters": {
            "contamination": result["contamination"],
            "n_estimators": 100,
        },
        "feature_cols": feature_cols,
        "mlflow_run_id": result["run_id"],
        "versioned_file": versioned_path.name,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVE] {version_str} (isolation_forest) → {current_path}")
    print(f"[SAVE] 백업: {versioned_path.name}")
    print(f"[SAVE] 메타데이터: {metadata_path}")


def main():
    print("=" * 60)
    print("환율 이상치 탐지 모델 학습 시작")
    print("=" * 60)

    rates, change_percents = load_data_from_db()
    X, feature_cols = make_features(rates, change_percents)
    print(f"[INFO] 전체 샘플 수: {len(X)}, 피처 수: {len(feature_cols)}")

    result = train_detector(X, feature_cols, contamination=0.05)

    print("\n" + "=" * 60)
    print(f"[DONE] 이상치 {result['n_anomalies']}건 탐지")
    print("=" * 60)

    save_model(result, feature_cols)


if __name__ == "__main__":
    main()