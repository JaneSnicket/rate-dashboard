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
    """최신 모델과 최신 환율 데이터를 불러와 내일의 등락을 예측합니다."""
    model_name = f"Exchange-Rate-Model-{currency}"
    model_uri = f"models:/{model_name}@champion"
    
    try:
        # MLflow 레지스트리에서 champion 모델 불러오기
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        return None, f"서비스 가능한(champion) 모델이 없습니다: {e}"

    # 2. 로컬 DB에서 오늘(가장 최근) 환율 데이터 1줄만 가져오기
    query = f"SELECT rate, change_percent FROM exchange_rates WHERE target_currency = '{currency}' ORDER BY collected_at DESC LIMIT 1"
    df_latest = pd.read_sql(query, engine)
    
    if df_latest.empty:
        return None, "환율 데이터가 부족하여 예측할 수 없음"

    # 3. 모델에 넣고 예측 수행 (0: 유지/하락, 1: 상승)
    pred = model.predict(df_latest)[0]
    result_text = "상승" if pred == 1 else "유지/하락"
    
    return result_text, "성공"