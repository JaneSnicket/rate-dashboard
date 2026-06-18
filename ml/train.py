import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

# 환경 변수 및 DB 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rate_dashboard.db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:6430") 

engine = create_engine(DATABASE_URL)

def load_and_preprocess_data(currency="KRW"):
    """DB에서 환율 데이터를 불러와 학습용 데이터로 전처리"""
    query = f"SELECT * FROM exchange_rates WHERE target_currency = '{currency}' ORDER BY collected_at ASC"
    df = pd.read_sql(query, engine)
    
    if len(df) < 10:
        return None, None
  
    df['next_rate'] = df['rate'].shift(-1)
    df = df.dropna() 
    
    # 내일 환율이 오늘보다 크면 1(상승), 아니면 0(하락)
    df['target'] = (df['next_rate'] > df['rate']).astype(int)
    
    # 학습에 사용할 변수(Features): 현재 환율, 전일 대비 변동률
    X = df[['rate', 'change_percent']]
    y = df['target']
    
    return X, y

def train_model(currency="KRW"):
    print(f"[{currency}] 데이터 로딩 및 전처리 시작...")
    X, y = load_and_preprocess_data(currency)
    
    if X is None:
        print("데이터가 충분하지 않음. 대시보드에서 '최신 환율 수집'을 여러 번 눌러 데이터를 쌓아야함.")
        return

    # 시계열 데이터이므로 순서를 섞지 않고(shuffle=False) 과거 데이터로 미래를 예측하도록 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # MLflow 서버 연결 및 실험 세팅
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Exchange-Rate-Prediction")

    with mlflow.start_run() as run:
        print("모델 학습 중")
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # 모델 성능 평가
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f" [{currency}] 모델 학습 완료. 테스트 정확도: {acc:.4f}")

        # MLflow에 파라미터 및 결과 기록
        mlflow.log_param("currency", currency)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        # 모델 파일(Artifact) 저장
        mlflow.sklearn.log_model(model, "model")
        print(f"MLflow Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model("KRW")