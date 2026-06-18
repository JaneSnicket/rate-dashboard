"""
환율 예측 API 라우터
- POST /predict/krw : KRW 환율 다음날 예측
- GET /predict/history : 과거 예측 이력 조회
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models import ExchangeRate, Prediction
from app.ml.registry import predictor_registry

logger = logging.getLogger("ml.predict")

router = APIRouter(prefix="/predict", tags=["predict"])

# 학습 시 사용한 피처
FEATURE_COLS = ["lag_1", "lag_3", "lag_7", "rolling_mean_7", "rolling_std_7"]
MIN_HISTORY = 7  


def build_features_from_db(db: Session) -> np.ndarray:
    """
    DB에서 KRW 환율 이력을 읽어 마지막 시점의 피처를 생성.
    학습 스크립트의 make_features와 동일한 로직.
    """
    rows = (
        db.query(ExchangeRate)
        .filter(ExchangeRate.target_currency == "KRW")
        .order_by(ExchangeRate.collected_at.asc())
        .all()
    )
    if len(rows) < MIN_HISTORY:
        raise HTTPException(
            status_code=400,
            detail=f"예측을 위한 데이터가 부족합니다. "
                   f"최소 {MIN_HISTORY}건 필요, 현재 {len(rows)}건. "
                   f"/rates/collect 를 여러 번 호출하여 데이터를 수집하세요.",
        )

    rates = [r.rate for r in rows]
    df = pd.DataFrame({"rate": rates})
    df["lag_1"] = df["rate"].shift(1)
    df["lag_3"] = df["rate"].shift(3)
    df["lag_7"] = df["rate"].shift(7)
    df["rolling_mean_7"] = df["rate"].rolling(7).mean()
    df["rolling_std_7"] = df["rate"].rolling(7).std()
    df = df.dropna()

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="피처 생성 가능한 시점이 없습니다. 데이터를 더 수집하세요.",
        )

    last_features = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    return last_features


@router.post("/krw")
def predict_krw_rate(db: Session = Depends(get_db)):
    """
    KRW 환율 다음날 예측.
    - 현재 운영 중인 predictor 모델 사용
    - 예측 결과를 DB에 저장 (운영 로그)
    """
    try:
        model = predictor_registry.load_current()
        metadata = predictor_registry.get_metadata()
    except FileNotFoundError as e:
        logger.error(f"모델 로딩 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="예측 모델이 학습되지 않았습니다. "
                   "python -m ml.train_predictor 를 먼저 실행하세요.",
        )

    # 피처 생성
    X = build_features_from_db(db)

    # 예측 수행
    try:
        predicted_rate = float(model.predict(X)[0])
    except Exception as e:
        logger.error(f"예측 실행 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"모델 예측 중 오류 발생: {str(e)}",
        )

    # 예측 결과 저장 (운영 로그)
    now = datetime.utcnow()
    predicted_for = now + timedelta(days=1)
    record = Prediction(
        target_currency="KRW",
        predicted_rate=predicted_rate,
        model_version=metadata.get("version", "unknown"),
        model_name=metadata.get("model_name", "unknown"),
        predicted_for=predicted_for,
        predicted_at=now,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        f"[PREDICT] currency=KRW, predicted_rate={predicted_rate:.4f}, "
        f"model={metadata.get('model_name')}:{metadata.get('version')}"
    )

    return {
        "id": record.id,
        "target_currency": "KRW",
        "predicted_rate": round(predicted_rate, 4),
        "predicted_for": predicted_for.isoformat(),
        "predicted_at": now.isoformat(),
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("version"),
    }


@router.get("/history")
def get_prediction_history(limit: int = 20, db: Session = Depends(get_db)):
    """과거 예측 이력 조회 (최신순)"""
    records = (
        db.query(Prediction)
        .filter(Prediction.target_currency == "KRW")
        .order_by(desc(Prediction.predicted_at))
        .limit(limit)
        .all()
    )
    if not records:
        return {"count": 0, "predictions": []}

    return {
        "count": len(records),
        "predictions": [
            {
                "id": r.id,
                "predicted_rate": r.predicted_rate,
                "actual_rate": r.actual_rate,
                "model_name": r.model_name,
                "model_version": r.model_version,
                "predicted_for": r.predicted_for.isoformat(),
                "predicted_at": r.predicted_at.isoformat(),
            }
            for r in records
        ],
    }