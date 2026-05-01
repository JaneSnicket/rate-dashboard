from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.database import get_db
from app.models import ExchangeRate
from app.collector import save_rates

router = APIRouter(prefix="/rates", tags=["rates"])

@router.post("/collect")
def collect_rates(db: Session = Depends(get_db)):
    """환율 데이터 수집 및 저장"""
    saved = save_rates(db)
    return {"message": f"{len(saved)}개 통화 데이터 저장 완료"}

@router.get("/latest")
def get_latest_rates(db: Session = Depends(get_db)):
    """최신 환율 조회"""
    currencies = ["KRW", "EUR", "JPY", "CNY", "GBP"]
    result = []

    for currency in currencies:
        rate = (
            db.query(ExchangeRate)
            .filter(ExchangeRate.target_currency == currency)
            .order_by(desc(ExchangeRate.collected_at))
            .first()
        )
        if rate:
            result.append({
                "currency": rate.target_currency,
                "rate": rate.rate,
                "change_percent": rate.change_percent,
                "collected_at": rate.collected_at
            })

    if not result:
        raise HTTPException(status_code=404, detail="데이터가 없습니다. /rates/collect 를 먼저 호출하세요.")

    return result

@router.get("/history/{currency}")
def get_rate_history(currency: str, limit: int = 30, db: Session = Depends(get_db)):
    """특정 통화의 이력 조회"""
    currency = currency.upper()
    rates = (
        db.query(ExchangeRate)
        .filter(ExchangeRate.target_currency == currency)
        .order_by(desc(ExchangeRate.collected_at))
        .limit(limit)
        .all()
    )

    if not rates:
        raise HTTPException(status_code=404, detail=f"{currency} 데이터가 없습니다.")

    return [
        {
            "currency": r.target_currency,
            "rate": r.rate,
            "change_percent": r.change_percent,
            "collected_at": r.collected_at
        }
        for r in rates
    ]