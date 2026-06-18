import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:6430")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def rollback_model(currency="KRW", target_version=1):
    """지정된 버전의 모델을 다시 champion으로 롤백"""
    client = MlflowClient()
    model_name = f"Exchange-Rate-Model-{currency}"
    
    try:
        # 특정 버전을 다시 champion으로 지정
        client.set_registered_model_alias(model_name, "champion", str(target_version))
        print(f"롤백 성공 {model_name}의 v{target_version} 모델이 다시 서비스(champion)에 반영됨")
    except Exception as e:
        print(f"롤백 실패: {e}")

if __name__ == "__main__":
    rollback_model("KRW", target_version=1)