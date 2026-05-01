import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models import ExchangeRate

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