"""
모델 레지스트리: 학습된 모델 로딩 / 버전 관리 / 롤백
- Predictor (환율 예측)와 Detector (이상치 탐지) 두 종류 모델 관리
- current.pkl을 운영 모델로 사용
- 버전 파일(v1_xxxxx.pkl, v2_xxxxx.pkl ...) 보관
- 롤백 시 지정된 버전을 current.pkl로 교체
"""
import json
import shutil
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


class ModelRegistry:
    """모델 종류별 레지스트리 (predictor / detector)"""

    def __init__(self, model_type: str):
        """
        Args:
            model_type: "predictor" 또는 "detector"
        """
        if model_type not in ("predictor", "detector"):
            raise ValueError(f"잘못된 model_type: {model_type}")
        self.model_type = model_type
        self.dir = MODELS_DIR / model_type
        self.current_path = self.dir / "current.pkl"
        self.metadata_path = self.dir / "metadata.json"

    # 모델 로딩
    def load_current(self):
        """현재 운영 모델 로드"""
        if not self.current_path.exists():
            raise FileNotFoundError(
                f"{self.model_type} 모델이 학습되지 않았습니다. "
                f"먼저 python -m ml.train_{self.model_type} 실행"
            )
        return joblib.load(self.current_path)

    def get_metadata(self) -> dict:
        """현재 운영 모델의 메타데이터 조회"""
        if not self.metadata_path.exists():
            return {"error": "메타데이터 없음. 모델을 먼저 학습하세요."}
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)


    # 버전 관리
    def list_versions(self) -> list:
        """저장된 모든 버전 파일 목록 (최신순)"""
        if not self.dir.exists():
            return []
        # v1_xxx.pkl, v2_xxx.pkl 형식 파일만 추출
        versions = [
            p for p in self.dir.glob("v*_*.pkl")
            if p.is_file()
        ]
        # 파일명에서 버전 번호 추출하여 정렬 (v10이 v2보다 뒤에 오도록)
        def version_num(p: Path) -> int:
            try:
                return int(p.stem.split("_")[0][1:])  # "v3_20260619" → 3
            except (ValueError, IndexError):
                return 0
        versions.sort(key=version_num, reverse=True)
        return [p.name for p in versions]

    def get_current_version(self) -> Optional[str]:
        """현재 운영 모델의 버전 문자열 (예: 'v3')"""
        meta = self.get_metadata()
        return meta.get("version") if "version" in meta else None
    # 롤백
    def rollback_to(self, version_filename: str) -> dict:
        """
        지정된 버전 파일을 current.pkl로 교체.
        
        Args:
            version_filename: "v2_20260619_104530.pkl" 형식의 파일명
        
        Returns:
            롤백 결과 메타데이터
        """
        target_path = self.dir / version_filename
        if not target_path.exists():
            raise FileNotFoundError(f"대상 버전 파일 없음: {version_filename}")

        # 롤백 전 현재 메타데이터 백업
        previous_version = self.get_current_version()

        # current.pkl을 새 버전으로 교체
        shutil.copy2(target_path, self.current_path)

        # 메타데이터 갱신 (롤백 이력 기록)
        version_str = version_filename.split("_")[0]  # "v2_20260619_..." → "v2"
        rollback_info = {
            "version": version_str,
            "model_name": self.get_metadata().get("model_name", "unknown"),
            "rolled_back_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "rolled_back_from": previous_version,
            "rolled_back_to_file": version_filename,
            "is_rollback": True,
        }

        # 기존 메타데이터에 롤백 정보 병합
        meta = self.get_metadata()
        if "error" not in meta:
            rollback_info["original_metrics"] = meta.get("metrics", {})
            rollback_info["feature_cols"] = meta.get("feature_cols", [])
            rollback_info["version_number"] = meta.get("version_number", 0)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(rollback_info, f, indent=2, ensure_ascii=False)

        return rollback_info

    def rollback_to_previous(self) -> dict:
        """가장 최근 직전 버전으로 롤백"""
        versions = self.list_versions()
        if len(versions) < 2:
            raise ValueError(
                f"롤백 가능한 이전 버전이 없습니다. "
                f"현재 저장된 버전: {len(versions)}개"
            )
        # 최신순 정렬돼 있으므로 [0]은 현재, [1]이 직전
        previous = versions[1]
        return self.rollback_to(previous)


# 싱글톤 인스턴스 
predictor_registry = ModelRegistry("predictor")
detector_registry = ModelRegistry("detector")