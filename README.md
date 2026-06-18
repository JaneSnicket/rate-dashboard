# 인공지능파이프라인 기말과제
## 인공지능학부 224643 제혜정

# 환율 대시보드 + ML (MLOps 파이프라인)

실시간 환율 데이터 수집/시각화 + **ML 기반 KRW 환율 예측 및 이상치 탐지** 대시보드
중간과제 DevOps 파이프라인을 확장하여 MLflow 기반 실험 관리, 모델 버전 관리, 롤백 기능까지 포함한 MLOps 파이프라인 구축

## 배포 URL
https://rate-dashboard-hyz4.onrender.com

## API 문서
https://rate-dashboard-hyz4.onrender.com/docs

## MLflow Tracking Server
로컬 환경에서 `mlflow ui` 또는 `docker compose up -d mlflow` 실행 후 http://localhost:5000 접속.
외부 노출이 필요한 경우 ngrok으로 5000번 포트를 터널링하여 사용 
## 기술 스택
- Backend: FastAPI
- Database: SQLite
- ML: scikit-learn (LinearRegression, Ridge, RandomForest, IsolationForest)
- 실험 관리: MLflow
- CI: GitHub Actions
- Container: Docker, docker-compose
- Deploy: Render
- 외부 노출: ngrok (MLflow)
## 주요 기능

### 서비스 기능 (중간과제부터 유지)
- 5개 통화(KRW/EUR/JPY/CNY/GBP) 실시간 환율 수집
- 통화별 최신 환율 카드 + 최대 30건 이력 차트

### ML 기능 (기말과제 추가)
- KRW 환율 예측 (다음날 환율 회귀 예측)
- KRW 환율 이상치 탐지 (급등락 자동 감지)
- 모델 버전 관리 및 롤백 (CLI)
- 운영 중 모델 정보 조회 API

## 디렉토리 구조

```
rate-dashboard/
├── app/                          # FastAPI 애플리케이션
│   ├── main.py
│   ├── database.py
│   ├── models.py                 # ExchangeRate + Prediction + Anomaly
│   ├── collector.py
│   ├── routers/
│   │   ├── rates.py              # 환율 API (중간과제)
│   │   ├── predict.py            # 예측 API (기말과제)
│   │   ├── anomaly.py            # 이상치 탐지 API (기말과제)
│   │   └── ml_info.py            # 모델 정보 API (기말과제)
│   └── ml/
│       └── registry.py           # 모델 레지스트리 + 롤백
├── ml/                           # 학습 스크립트 (기말과제)
│   ├── train_predictor.py
│   ├── train_detector.py
│   └── rollback_model.py
├── models/                       # 학습된 모델 파일 + 버전 백업
│   ├── predictor/
│   └── detector/
├── tests/                        # 50개 테스트 (중간 19 → 기말 50)
├── static/index.html             # 대시보드 UI
├── .github/workflows/
│   ├── ci.yml                    # 테스트 + 빌드 자동화
│   └── retrain.yml               # 모델 재학습 (수동/주간)
├── Dockerfile
├── docker-compose.yml            # web + MLflow 동시 기동
└── requirements.txt
```

## 실행 방법

### 로컬 실행
```bash
uvicorn app.main:app --reload
```

### Docker 실행
```bash
docker compose up --build
```
- 웹: http://localhost:8000
- MLflow UI: http://localhost:5000

### 테스트 실행
```bash
pytest tests/ -v
```
총 50개 테스트 (test_api 8 + test_collector 7 + test_models 10 + test_ml_registry 12 + test_ml_api 13)

## ML 모델 학습

```bash
# 환율 예측 모델 학습 (3개 알고리즘 비교 → 최고 성능 자동 채택)
python -m ml.train_predictor

# 이상치 탐지 모델 학습 (IsolationForest)
python -m ml.train_detector
```

학습 시 자동 처리
- MLflow에 parameter / metric / artifact / model 기록
- 새 버전 자동 부여 (v1, v2, v3, …)
- 운영 모델(`current.pkl`) + 버전별 백업(`v{N}_*.pkl`) 저장
- 버전별 메타데이터(`v{N}_meta.json`) 저장 (롤백 시 정보 복원용)

## MLflow UI 실행

```bash
# 로컬 직접 실행
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Docker로 실행
docker compose up -d mlflow
```

## 모델 버전 관리 및 롤백

```bash
# 버전 목록 조회
python -m ml.rollback_model predictor --list

# 특정 버전으로 롤백
python -m ml.rollback_model predictor v2

# 직전 버전으로 자동 롤백
python -m ml.rollback_model predictor
```

롤백 시 자동 처리
- `current.pkl`을 대상 버전으로 교체
- `metadata.json`에 롤백 이력 기록 (`is_rollback`, `rolled_back_from`, `rolled_back_at`)

## CI/CD 파이프라인

### `ci.yml` (자동 실행 - main 브랜치 push 시)
1. test job: pytest 50개 실행 + 커버리지 리포트
2. ml_smoke_test job: 학습 스크립트 동작 검증
3. build job: Docker 이미지 빌드 검증

### `retrain.yml` (모델 재학습 - 수동/주간)
- 수동 트리거: GitHub Actions → Run workflow
- 자동 스케줄: 매주 월요일 09:00 KST
- 학습된 모델 30일간 아티팩트로 보관

### Render 자동 배포
main 브랜치 push 감지 시 자동 재배포


## 주요 API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 헬스체크 |
| POST | `/rates/collect` | 환율 수집 |
| GET | `/rates/latest` | 최신 환율 |
| GET | `/rates/history/{currency}` | 통화별 이력 |
| POST | `/predict/krw` | KRW 다음날 예측 |
| GET | `/predict/history` | 예측 이력 |
| POST | `/anomaly/check` | 이상치 탐지 |
| GET | `/anomaly/history` | 탐지 이력 |
| GET | `/ml/model-info` | 현재 운영 모델 정보 |
| GET | `/ml/versions/{model_type}` | 저장된 버전 목록 |