from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.database import engine, Base
from app.routers import rates

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