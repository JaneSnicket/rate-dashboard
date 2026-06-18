"""
환율 이상치 탐지 API 라우터
- POST /anomaly/check : 가장 최근 KRW 환율의 이상치 여부 판정
- GET /anomaly/history : 탐지된 이상치 이력 조회
"""
import logging
import numpy as np
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import ExchangeRate, Anomaly
from app.ml.registry import detector_registry

logger = logging.getLogger("ml.anomaly")

router = APIRouter(prefix="/anomaly", tags=["anomaly"])

FEATURE_COLS = ["rate", "change_percent", "abs_change"]


@router.post("/check")
def check_latest_anomaly(db: Session = Depends(get_db)):
    """
    가장 최근 수집된 KRW 환율이 이상치인지 판정.
    결과를 DB에 저장 (운영 로그).
    """
    try:
        model = detector_registry.load_current()
        metadata = detector_registry.get_metadata()
    except FileNotFoundError as e:
        logger.error(f"이상치 탐지 모델 로딩 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="이상치 탐지 모델이 학습되지 않았습니다. "
                   "python -m ml.train_detector 를 먼저 실행하세요.",
        )

    # 가장 최근 KRW 환율 조회
    latest = (
        db.query(ExchangeRate)
        .filter(ExchangeRate.target_currency == "KRW")
        .order_by(desc(ExchangeRate.collected_at))
        .first()
    )
    if not latest:
        raise HTTPException(
            status_code=400,
            detail="KRW 환율 데이터가 없습니다. /rates/collect 를 먼저 호출하세요.",
        )

    # 피처 구성
    X = np.array([[latest.rate, latest.change_percent, abs(latest.change_percent)]])

    # 예측
    try:
        pred = int(model.predict(X)[0])         
        score = float(model.score_samples(X)[0])  
    except Exception as e:
        logger.error(f"이상치 탐지 실행 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"모델 추론 중 오류 발생: {str(e)}",
        )

    is_anomaly = (pred == -1)

    # 결과 저장 
    record = Anomaly(
        target_currency="KRW",
        rate=latest.rate,
        change_percent=latest.change_percent,
        anomaly_score=score,
        is_anomaly=is_anomaly,
        model_version=metadata.get("version", "unknown"),
        detected_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        f"[ANOMALY] currency=KRW, rate={latest.rate}, "
        f"change={latest.change_percent}%, score={score:.4f}, "
        f"is_anomaly={is_anomaly}, model_version={metadata.get('version')}"
    )

    return {
        "id": record.id,
        "target_currency": "KRW",
        "rate": latest.rate,
        "change_percent": latest.change_percent,
        "anomaly_score": round(score, 4),
        "is_anomaly": is_anomaly,
        "model_version": metadata.get("version"),
        "detected_at": record.detected_at.isoformat(),
    }


@router.get("/history")
def get_anomaly_history(only_anomalies: bool = False, limit: int = 20,
                       db: Session = Depends(get_db)):
    """
    이상치 탐지 이력 조회.
    - only_anomalies=true 면 is_anomaly=True인 것만 반환
    """
    q = db.query(Anomaly).filter(Anomaly.target_currency == "KRW")
    if only_anomalies:
        q = q.filter(Anomaly.is_anomaly == True)  # noqa: E712
    records = q.order_by(desc(Anomaly.detected_at)).limit(limit).all()

    return {
        "count": len(records),
        "only_anomalies": only_anomalies,
        "results": [
            {
                "id": r.id,
                "rate": r.rate,
                "change_percent": r.change_percent,
                "anomaly_score": r.anomaly_score,
                "is_anomaly": r.is_anomaly,
                "model_version": r.model_version,
                "detected_at": r.detected_at.isoformat(),
            }
            for r in records
        ],
    }