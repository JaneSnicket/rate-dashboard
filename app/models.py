from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.database import Base

class ExchangeRate(Base):
    __tablename__ = "exchange_rates"

    id = Column(Integer, primary_key=True, index=True)
    base_currency = Column(String, default="USD")
    target_currency = Column(String, nullable=False)
    rate = Column(Float, nullable=False)
    change_percent = Column(Float, default=0.0)
    collected_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ExchangeRate {self.base_currency}/{self.target_currency} = {self.rate}>"