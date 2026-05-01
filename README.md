# 인공지능파이프라인 중간과제
## 인공지능학부 224643 제혜정

# 환율 대시보드

실시간 환율 데이터 수집 및 시각화 대시보드

## 배포 URL
https://rate-dashboard-hyz4.onrender.com

## 기술 스택
- Backend: FastAPI
- Database: SQLite
- CI: GitHub Actions
- Container: Docker
- Deploy: Render

## 실행 방법
# 로컬 실행
uvicorn app.main:app --reload

# Docker 실행
docker compose up --build

# 테스트 실행
pytest tests/ -v