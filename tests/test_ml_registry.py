import json
import joblib
import pytest
from pathlib import Path
from unittest.mock import patch
from app.ml.registry import ModelRegistry


@pytest.fixture
def temp_models_dir(tmp_path, monkeypatch):
    """임시 모델 디렉토리 생성 (실제 운영 모델 건드리지 않음)"""
    fake_root = tmp_path / "models"
    fake_root.mkdir()
    (fake_root / "predictor").mkdir()
    (fake_root / "detector").mkdir()

    # MODELS_DIR을 임시 경로로 교체
    monkeypatch.setattr("app.ml.registry.MODELS_DIR", fake_root)
    return fake_root


def _create_dummy_model_file(path: Path, model_obj=None):
    """더미 모델 파일 생성"""
    if model_obj is None:
        model_obj = {"dummy": "model", "name": path.stem}
    joblib.dump(model_obj, path)


def _create_metadata(path: Path, version_str: str, version_num: int):
    """더미 메타데이터 생성"""
    meta = {
        "version": version_str,
        "version_number": version_num,
        "model_name": "ridge",
        "metrics": {"rmse": 5.0, "mae": 4.0, "r2": -0.5},
        "feature_cols": ["lag_1", "lag_3"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


# 기본 동작 테스트

def test_invalid_model_type():
    """잘못된 model_type 입력 시 에러"""
    with pytest.raises(ValueError):
        ModelRegistry("invalid_type")


def test_load_current_when_no_model(temp_models_dir):
    """모델 없을 때 FileNotFoundError"""
    reg = ModelRegistry("predictor")
    with pytest.raises(FileNotFoundError):
        reg.load_current()


def test_load_current_success(temp_models_dir):
    """current.pkl 로딩 성공"""
    reg = ModelRegistry("predictor")
    _create_dummy_model_file(reg.current_path, {"name": "current"})
    loaded = reg.load_current()
    assert loaded["name"] == "current"


def test_get_metadata_when_none(temp_models_dir):
    """메타데이터 없을 때 error 키 반환"""
    reg = ModelRegistry("predictor")
    meta = reg.get_metadata()
    assert "error" in meta


def test_get_metadata_success(temp_models_dir):
    """메타데이터 조회 성공"""
    reg = ModelRegistry("predictor")
    _create_metadata(reg.metadata_path, "v3", 3)
    meta = reg.get_metadata()
    assert meta["version"] == "v3"
    assert meta["version_number"] == 3


# 버전 관리 테스트

def test_list_versions_empty(temp_models_dir):
    """버전 파일 없을 때 빈 리스트"""
    reg = ModelRegistry("predictor")
    assert reg.list_versions() == []


def test_list_versions_sorted_desc(temp_models_dir):
    """버전 파일 최신순 정렬"""
    reg = ModelRegistry("predictor")
    _create_dummy_model_file(reg.dir / "v1_20260101_000000.pkl")
    _create_dummy_model_file(reg.dir / "v3_20260103_000000.pkl")
    _create_dummy_model_file(reg.dir / "v2_20260102_000000.pkl")
    _create_dummy_model_file(reg.dir / "v10_20260110_000000.pkl")
    versions = reg.list_versions()
    # v10이 v2보다 앞에 와야 함 (숫자 정렬)
    assert versions[0].startswith("v10_")
    assert versions[1].startswith("v3_")
    assert versions[2].startswith("v2_")
    assert versions[3].startswith("v1_")


def test_get_current_version(temp_models_dir):
    """현재 버전 조회"""
    reg = ModelRegistry("predictor")
    _create_metadata(reg.metadata_path, "v5", 5)
    assert reg.get_current_version() == "v5"


# 롤백 테스트 

def test_rollback_to_specific_version(temp_models_dir):
    """특정 버전으로 롤백"""
    reg = ModelRegistry("predictor")
    # 두 버전 준비
    _create_dummy_model_file(reg.dir / "v1_20260101.pkl", {"version": "v1"})
    _create_dummy_model_file(reg.dir / "v2_20260102.pkl", {"version": "v2"})
    # 현재는 v2
    _create_dummy_model_file(reg.current_path, {"version": "v2"})
    _create_metadata(reg.metadata_path, "v2", 2)

    # v1으로 롤백
    result = reg.rollback_to("v1_20260101.pkl")

    # 검증: current.pkl이 v1 내용으로 교체됨
    current = joblib.load(reg.current_path)
    assert current["version"] == "v1"
    assert result["version"] == "v1"
    assert result["rolled_back_from"] == "v2"
    assert result["is_rollback"] is True


def test_rollback_to_previous(temp_models_dir):
    """직전 버전으로 자동 롤백"""
    reg = ModelRegistry("predictor")
    _create_dummy_model_file(reg.dir / "v1_20260101.pkl", {"version": "v1"})
    _create_dummy_model_file(reg.dir / "v2_20260102.pkl", {"version": "v2"})
    _create_dummy_model_file(reg.dir / "v3_20260103.pkl", {"version": "v3"})
    _create_dummy_model_file(reg.current_path, {"version": "v3"})
    _create_metadata(reg.metadata_path, "v3", 3)

    result = reg.rollback_to_previous()

    current = joblib.load(reg.current_path)
    assert current["version"] == "v2"
    assert result["version"] == "v2"


def test_rollback_fails_when_only_one_version(temp_models_dir):
    """버전 1개뿐일 때 롤백 실패"""
    reg = ModelRegistry("predictor")
    _create_dummy_model_file(reg.dir / "v1_20260101.pkl", {"version": "v1"})
    _create_dummy_model_file(reg.current_path, {"version": "v1"})

    with pytest.raises(ValueError):
        reg.rollback_to_previous()


def test_rollback_fails_for_missing_file(temp_models_dir):
    """존재하지 않는 파일로 롤백 시도 시 에러"""
    reg = ModelRegistry("predictor")
    with pytest.raises(FileNotFoundError):
        reg.rollback_to("v999_nonexistent.pkl")