import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import ExchangeRate

load_dotenv()

API_KEY = os.getenv("EXCHANGE_API_KEY", "")
BASE_URL = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/USD"

TARGET_CURRENCIES = ["KRW", "EUR", "JPY", "CNY", "GBP"]

def fetch_exchange_rates() -> dict:
    """외부 API에서 환율 데이터를 가져옴"""
    dummy_data = {
        "KRW": 1350.0,
        "EUR": 0.92,
        "JPY": 154.5,
        "CNY": 7.24,
        "GBP": 0.79
    }

    if not API_KEY or API_KEY == "your_api_key_here" or API_KEY == "test_key":
        return dummy_data

    try:
        response = requests.get(BASE_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            currency: data["conversion_rates"][currency]
            for currency in TARGET_CURRENCIES
            if currency in data["conversion_rates"]
        }
    except requests.exceptions.RequestException:
        return dummy_data

def calculate_change_percent(current_rate: float, previous_rate: float) -> float:
    """전일 대비 등락률 계산"""
    if previous_rate == 0:
        return 0.0
    return round((current_rate - previous_rate) / previous_rate * 100, 4)

def save_rates(db: Session) -> list:
    """환율 데이터를 수집하고 DB에 저장"""
    rates = fetch_exchange_rates()
    saved = []

    for currency, rate in rates.items():
        previous = (
            db.query(ExchangeRate)
            .filter(ExchangeRate.target_currency == currency)
            .order_by(ExchangeRate.collected_at.desc())
            .first()
        )

        change = calculate_change_percent(rate, previous.rate) if previous else 0.0

        new_rate = ExchangeRate(
            base_currency="USD",
            target_currency=currency,
            rate=rate,
            change_percent=change,
            collected_at=datetime.utcnow()
        )
        db.add(new_rate)
        saved.append(new_rate)

    db.commit()
    return saved