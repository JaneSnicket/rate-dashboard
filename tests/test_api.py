import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db

TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

def test_health_check():
    """서버 상태 확인"""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_collect_rates():
    """/rates/collect 호출 시 데이터 저장"""
    res = client.post("/rates/collect")
    assert res.status_code == 200
    assert "저장 완료" in res.json()["message"]

def test_get_latest_rates_after_collect():
    """수집 후 최신 환율 조회"""
    client.post("/rates/collect")
    res = client.get("/rates/latest")
    assert res.status_code == 200
    data = res.json()
    assert len(data) > 0
    assert "currency" in data[0]
    assert "rate" in data[0]
    assert "change_percent" in data[0]

def test_get_latest_rates_empty():
    """데이터 없을 때 404 반환"""
    res = client.get("/rates/latest")
    assert res.status_code == 404

def test_get_history_after_collect():
    """수집 후 이력 조회"""
    client.post("/rates/collect")
    res = client.get("/rates/history/KRW")
    assert res.status_code == 200
    data = res.json()
    assert len(data) > 0
    assert data[0]["currency"] == "KRW"

def test_get_history_invalid_currency():
    """존재하지 않는 통화 조회 시 404"""
    res = client.get("/rates/history/XYZ")
    assert res.status_code == 404

def test_get_history_limit():
    """limit 파라미터 동작 확인"""
    for _ in range(5):
        client.post("/rates/collect")
    res = client.get("/rates/history/KRW?limit=2")
    assert res.status_code == 200
    assert len(res.json()) <= 2

def test_currency_uppercase():
    """소문자 통화 코드도 처리"""
    client.post("/rates/collect")
    res = client.get("/rates/history/krw")
    assert res.status_code == 200