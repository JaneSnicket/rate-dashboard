import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models import ExchangeRate, Prediction, Anomaly

TEST_DATABASE_URL = "sqlite:///./test_models.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=engine)

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def get_session():
    return TestingSessionLocal()

# ExchangeRate 테스트 (중간과제 유지)

def test_create_exchange_rate():
    """환율 데이터 DB 저장"""
    db = get_session()
    rate = ExchangeRate(target_currency="KRW", rate=1350.0)
    db.add(rate)
    db.commit()
    db.refresh(rate)
    assert rate.id is not None
    assert rate.target_currency == "KRW"
    db.close()


def test_default_base_currency():
    """기본 기준 통화가 USD인지 확인"""
    db = get_session()
    rate = ExchangeRate(target_currency="EUR", rate=0.92)
    db.add(rate)
    db.commit()
    db.refresh(rate)
    assert rate.base_currency == "USD"
    db.close()


def test_query_by_currency():
    """통화 코드로 조회"""
    db = get_session()
    db.add(ExchangeRate(target_currency="JPY", rate=154.5))
    db.add(ExchangeRate(target_currency="KRW", rate=1350.0))
    db.commit()
    result = db.query(ExchangeRate).filter_by(target_currency="JPY").first()
    assert result is not None
    assert result.rate == 154.5
    db.close()


def test_multiple_records():
    """여러 데이터 저장 및 조회"""
    db = get_session()
    for rate in [1350.0, 1355.0, 1360.0]:
        db.add(ExchangeRate(target_currency="KRW", rate=rate))
    db.commit()
    results = db.query(ExchangeRate).filter_by(target_currency="KRW").all()
    assert len(results) == 3
    db.close()

# Prediction 테스트 (기말과제 추가)

def test_create_prediction():
    """예측 결과 저장 테스트"""
    db = get_session()
    pred = Prediction(
        target_currency="KRW",
        predicted_rate=1380.5,
        model_version="v1",
        model_name="ridge",
        predicted_for=datetime.utcnow() + timedelta(days=1),
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    assert pred.id is not None
    assert pred.target_currency == "KRW"
    assert pred.actual_rate is None
    db.close()


def test_prediction_default_currency():
    """예측의 기본 통화가 KRW인지 확인"""
    db = get_session()
    pred = Prediction(
        predicted_rate=1380.5,
        model_version="v1",
        model_name="ridge",
        predicted_for=datetime.utcnow(),
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    assert pred.target_currency == "KRW"
    db.close()


def test_query_predictions_by_model_version():
    """모델 버전별 예측 조회 (롤백 시나리오)"""
    db = get_session()
    for v in ["v1", "v1", "v2"]:
        db.add(Prediction(
            predicted_rate=1380.0,
            model_version=v,
            model_name="ridge",
            predicted_for=datetime.utcnow(),
        ))
    db.commit()
    v1_preds = db.query(Prediction).filter_by(model_version="v1").all()
    assert len(v1_preds) == 2
    db.close()

# Anomaly 테스트 (기말과제 추가)

def test_create_anomaly():
    """이상치 탐지 결과 저장 테스트"""
    db = get_session()
    anomaly = Anomaly(
        target_currency="KRW",
        rate=1500.0,
        change_percent=8.5,
        anomaly_score=-0.45,
        is_anomaly=True,
        model_version="v1",
    )
    db.add(anomaly)
    db.commit()
    db.refresh(anomaly)
    assert anomaly.id is not None
    assert anomaly.is_anomaly is True
    db.close()


def test_query_only_anomalies():
    """is_anomaly=True인 것만 조회"""
    db = get_session()
    db.add(Anomaly(rate=1350.0, change_percent=0.1, anomaly_score=0.2,
                   is_anomaly=False, model_version="v1"))
    db.add(Anomaly(rate=1500.0, change_percent=8.5, anomaly_score=-0.45,
                   is_anomaly=True, model_version="v1"))
    db.commit()
    results = db.query(Anomaly).filter_by(is_anomaly=True).all()
    assert len(results) == 1
    assert results[0].rate == 1500.0
    db.close()


def test_anomaly_score_range():
    """이상치 점수가 음수일 수 있는지 (IsolationForest 특성)"""
    db = get_session()
    anomaly = Anomaly(
        rate=1500.0,
        change_percent=8.5,
        anomaly_score=-0.45,
        is_anomaly=True,
        model_version="v1",
    )
    db.add(anomaly)
    db.commit()
    db.refresh(anomaly)
    assert anomaly.anomaly_score < 0
    db.close()