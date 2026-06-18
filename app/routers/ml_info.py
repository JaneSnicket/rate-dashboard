"""
운영 중인 ML 모델 정보 조회 API
- GET /ml/model-info : 현재 운영 모델 메타데이터
- GET /ml/versions/{model_type} : 저장된 버전 목록
"""
from fastapi import APIRouter, HTTPException
from app.ml.registry import predictor_registry, detector_registry

router = APIRouter(prefix="/ml", tags=["ml-info"])


@router.get("/model-info")
def get_model_info():
    """현재 운영 중인 모든 모델의 메타데이터"""
    return {
        "predictor": predictor_registry.get_metadata(),
        "detector": detector_registry.get_metadata(),
    }


@router.get("/versions/{model_type}")
def list_model_versions(model_type: str):
    """저장된 모델 버전 목록 (롤백 가능한 버전 확인용)"""
    if model_type == "predictor":
        registry = predictor_registry
    elif model_type == "detector":
        registry = detector_registry
    else:
        raise HTTPException(
            status_code=400,
            detail=f"잘못된 model_type: {model_type}. predictor 또는 detector 만 가능",
        )

    return {
        "model_type": model_type,
        "current_version": registry.get_current_version(),
        "available_versions": registry.list_versions(),
    }