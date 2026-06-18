"""
ML API 라우터 통합 테스트
- predict / anomaly / ml-info 엔드포인트
"""
import shutil
import joblib
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest

from app.main import app
from app.database import Base, get_db
import app.ml.registry as registry_module


TEST_DATABASE_URL = "sqlite:///./test_ml.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def setup_db_and_models(tmp_path, monkeypatch):
    """매 테스트마다 DB + 임시 모델 디렉토리 초기화"""
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)

    # 임시 모델 디렉토리 생성
    fake_models = tmp_path / "models"
    fake_models.mkdir()
    (fake_models / "predictor").mkdir()
    (fake_models / "detector").mkdir()

    # registry 모듈의 MODELS_DIR 교체 + 기존 인스턴스도 다시 만들어줌
    monkeypatch.setattr(registry_module, "MODELS_DIR", fake_models)
    # predictor_registry, detector_registry 객체도 새 경로로 재생성
    new_predictor = registry_module.ModelRegistry("predictor")
    new_detector = registry_module.ModelRegistry("detector")
    monkeypatch.setattr(registry_module, "predictor_registry", new_predictor)
    monkeypatch.setattr(registry_module, "detector_registry", new_detector)

    # router에서 직접 import한 객체도 교체
    import app.routers.predict as predict_router
    import app.routers.anomaly as anomaly_router
    import app.routers.ml_info as ml_info_router
    monkeypatch.setattr(predict_router, "predictor_registry", new_predictor)
    monkeypatch.setattr(anomaly_router, "detector_registry", new_detector)
    monkeypatch.setattr(ml_info_router, "predictor_registry", new_predictor)
    monkeypatch.setattr(ml_info_router, "detector_registry", new_detector)

    yield {"models_dir": fake_models, "predictor": new_predictor, "detector": new_detector}

    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.pop(get_db, None)


def _install_dummy_predictor(registry, models_dir):
    """더미 예측 모델 설치"""
    import numpy as np
    # 5개 피처를 받는 간단한 Ridge 모델
    X = np.array([[1350, 1340, 1330, 1345, 5.0]] * 20)
    y = np.array([1355] * 20)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    joblib.dump(model, registry.current_path)

    import json
    meta = {
        "version": "v1",
        "version_number": 1,
        "model_name": "ridge",
        "trained_at": "20260619_120000",
        "metrics": {"rmse": 5.0, "mae": 4.0, "r2": -0.5},
        "feature_cols": ["lag_1", "lag_3", "lag_7", "rolling_mean_7", "rolling_std_7"],
    }
    with open(registry.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _install_dummy_detector(registry, models_dir):
    """더미 이상치 탐지 모델 설치"""
    import numpy as np
    X = np.random.RandomState(42).normal(0, 1, size=(100, 3))
    model = IsolationForest(contamination=0.05, random_state=42, n_estimators=10)
    model.fit(X)
    joblib.dump(model, registry.current_path)

    import json
    meta = {
        "version": "v1",
        "version_number": 1,
        "model_name": "isolation_forest",
        "trained_at": "20260619_120000",
        "metrics": {"n_anomalies_detected": 5, "anomaly_ratio": 0.05},
        "feature_cols": ["rate", "change_percent", "abs_change"],
    }
    with open(registry.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


client = TestClient(app)


# /ml/model-info 테스트

def test_model_info_when_no_models(setup_db_and_models):
    """모델 학습 안 됐을 때 model-info 응답"""
    res = client.get("/ml/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "predictor" in data
    assert "detector" in data
    # 메타데이터 없으면 error 키가 들어있음
    assert "error" in data["predictor"]


def test_model_info_with_models(setup_db_and_models):
    """모델 설치 후 model-info 응답"""
    ctx = setup_db_and_models
    _install_dummy_predictor(ctx["predictor"], ctx["models_dir"])
    _install_dummy_detector(ctx["detector"], ctx["models_dir"])

    res = client.get("/ml/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["predictor"]["version"] == "v1"
    assert data["predictor"]["model_name"] == "ridge"
    assert data["detector"]["model_name"] == "isolation_forest"


def test_ml_versions_endpoint(setup_db_and_models):
    """버전 목록 API"""
    res = client.get("/ml/versions/predictor")
    assert res.status_code == 200
    data = res.json()
    assert data["model_type"] == "predictor"
    assert "available_versions" in data


def test_ml_versions_invalid_type(setup_db_and_models):
    """잘못된 model_type 400 반환"""
    res = client.get("/ml/versions/invalid")
    assert res.status_code == 400


# /predict/krw 테스트

def test_predict_when_model_not_trained(setup_db_and_models):
    """모델 없을 때 503"""
    res = client.post("/predict/krw")
    assert res.status_code == 503


def test_predict_when_data_insufficient(setup_db_and_models):
    """데이터 부족 시 400"""
    ctx = setup_db_and_models
    _install_dummy_predictor(ctx["predictor"], ctx["models_dir"])
    res = client.post("/predict/krw")
    assert res.status_code == 400  # 환율 데이터 없음


def test_predict_success(setup_db_and_models):
    """정상 예측 시나리오"""
    ctx = setup_db_and_models
    _install_dummy_predictor(ctx["predictor"], ctx["models_dir"])
    # 환율 데이터 충분히 수집 (7건 이상)
    for _ in range(10):
        client.post("/rates/collect")
    res = client.post("/predict/krw")
    assert res.status_code == 200
    data = res.json()
    assert data["target_currency"] == "KRW"
    assert "predicted_rate" in data
    assert data["model_name"] == "ridge"
    assert data["model_version"] == "v1"


def test_predict_history_empty(setup_db_and_models):
    """예측 이력 없을 때"""
    res = client.get("/predict/history")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 0


def test_predict_saves_history(setup_db_and_models):
    """예측 실행 → 이력 저장 확인"""
    ctx = setup_db_and_models
    _install_dummy_predictor(ctx["predictor"], ctx["models_dir"])
    for _ in range(10):
        client.post("/rates/collect")
    client.post("/predict/krw")
    res = client.get("/predict/history")
    data = res.json()
    assert data["count"] >= 1



# /anomaly/check 테스트
def test_anomaly_when_model_not_trained(setup_db_and_models):
    """모델 없을 때 503"""
    res = client.post("/anomaly/check")
    assert res.status_code == 503


def test_anomaly_when_no_data(setup_db_and_models):
    """환율 데이터 없을 때 400"""
    ctx = setup_db_and_models
    _install_dummy_detector(ctx["detector"], ctx["models_dir"])
    res = client.post("/anomaly/check")
    assert res.status_code == 400


def test_anomaly_check_success(setup_db_and_models):
    """정상 이상치 탐지 시나리오"""
    ctx = setup_db_and_models
    _install_dummy_detector(ctx["detector"], ctx["models_dir"])
    client.post("/rates/collect")
    res = client.post("/anomaly/check")
    assert res.status_code == 200
    data = res.json()
    assert data["target_currency"] == "KRW"
    assert "anomaly_score" in data
    assert "is_anomaly" in data
    assert data["model_version"] == "v1"


def test_anomaly_history(setup_db_and_models):
    """이상치 이력 조회"""
    ctx = setup_db_and_models
    _install_dummy_detector(ctx["detector"], ctx["models_dir"])
    client.post("/rates/collect")
    client.post("/anomaly/check")
    res = client.get("/anomaly/history")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 1


def test_anomaly_history_only_anomalies_filter(setup_db_and_models):
    """only_anomalies 필터 동작 확인"""
    res = client.get("/anomaly/history?only_anomalies=true")
    assert res.status_code == 200
    data = res.json()
    assert data["only_anomalies"] is True