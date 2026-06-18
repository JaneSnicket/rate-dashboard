import os
import pandas as pd
import mlflow
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 환경 변수 설정
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rate_dashboard.db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:6430")

engine = create_engine(DATABASE_URL)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def predict_tomorrow(currency="KRW"):
    """최신 모델과 최신 환율 데이터를 불러와 내일의 등락 예측"""
    # 1. MLflow에서 해당 통화의 최신 '1세대' 모델 불러오기
    try:
        experiment = mlflow.get_experiment_by_name("Exchange-Rate-Prediction")
        if not experiment: 
            return None, "실험 데이터 없음"
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty: 
            return None, "학습된 모델 없음"
        currency_runs = runs[runs["params.currency"] == currency]
        if currency_runs.empty: 
            return None, f"{currency} 통화에 대한 모델이 없음"
        
        latest_run_id = currency_runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/model"
        
        # 모델 메모리에 로드
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        return None, f"모델 로딩 에러: {e}"

    # 2. 로컬 DB에서 오늘(가장 최근) 환율 데이터 1줄만 가져오기
    query = f"SELECT rate, change_percent FROM exchange_rates WHERE target_currency = '{currency}' ORDER BY collected_at DESC LIMIT 1"
    df_latest = pd.read_sql(query, engine)
    
    if df_latest.empty:
        return None, "환율 데이터가 부족하여 예측할 수 없음"

    # 3. 모델에 넣고 예측 수행 (0: 유지/하락, 1: 상승)
    pred = model.predict(df_latest)[0]
    result_text = "상승" if pred == 1 else "유지/하락"
    
    return result_text, "성공"