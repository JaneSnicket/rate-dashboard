from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.database import engine, Base
from app.routers import rates
from app.ai_service import predict_tomorrow

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="환율 대시보드 API",
    description="실시간 환율 데이터 수집 및 조회 API",
    version="1.0.0"
)

app.include_router(rates.router)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/predict/{currency}")
def get_prediction(currency: str):
    """지정된 통화의 내일 환율 등락 예측 결과를 반환합니다."""
    result, msg = predict_tomorrow(currency)
    
    if result:
        return {
            "target_currency": currency, 
            "prediction": result,
            "status": "success"
        }
    else:
        return {
            "target_currency": currency, 
            "error_message": msg,
            "status": "error"
        }