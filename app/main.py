import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.database import engine, Base
from app.routers import rates, predict, anomaly, ml_info

# 로깅 설정 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="환율 대시보드 API + ML",
    description="실시간 환율 데이터 수집/조회 + ML 기반 예측 및 이상치 탐지",
    version="2.0.0",
)

# 라우터 등록
app.include_router(rates.router)
app.include_router(predict.router)
app.include_router(anomaly.router)
app.include_router(ml_info.router)

# 정적 파일
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}