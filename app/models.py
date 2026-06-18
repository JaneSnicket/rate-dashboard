from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime
from app.database import Base


class ExchangeRate(Base):
    """환율 원본 데이터 (중간과제 유지)"""
    __tablename__ = "exchange_rates"

    id = Column(Integer, primary_key=True, index=True)
    base_currency = Column(String, default="USD")
    target_currency = Column(String, nullable=False)
    rate = Column(Float, nullable=False)
    change_percent = Column(Float, default=0.0)
    collected_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ExchangeRate {self.base_currency}/{self.target_currency} = {self.rate}>"


class Prediction(Base):
    """환율 예측 결과 (ML)"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    target_currency = Column(String, nullable=False, default="KRW")
    predicted_rate = Column(Float, nullable=False)
    actual_rate = Column(Float, nullable=True)
    model_version = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    predicted_for = Column(DateTime, nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.target_currency} {self.predicted_rate} by {self.model_name}:{self.model_version}>"


class Anomaly(Base):
    """이상치 탐지 결과 (ML)"""
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    target_currency = Column(String, nullable=False, default="KRW")
    rate = Column(Float, nullable=False)
    change_percent = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    model_version = Column(String, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Anomaly {self.target_currency} score={self.anomaly_score:.3f} is_anomaly={self.is_anomaly}>"