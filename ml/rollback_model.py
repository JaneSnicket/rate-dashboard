"""
모델 롤백 CLI 스크립트

사용법:
  python -m ml.rollback_model predictor          # 직전 버전으로 롤백
  python -m ml.rollback_model predictor v2       # v2로 롤백
  python -m ml.rollback_model predictor --list   # 버전 목록 조회
"""
import sys
from app.ml.registry import ModelRegistry


def main():
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python -m ml.rollback_model <predictor|detector> [--list | <version>]")
        print("")
        print("예시:")
        print("  python -m ml.rollback_model predictor --list")
        print("  python -m ml.rollback_model predictor          # 직전 버전으로 롤백")
        print("  python -m ml.rollback_model predictor v2       # v2로 롤백")
        sys.exit(1)

    model_type = sys.argv[1]
    if model_type not in ("predictor", "detector"):
        print(f"[ERROR] 잘못된 model_type: {model_type}")
        print("predictor 또는 detector 중 선택하세요.")
        sys.exit(1)

    registry = ModelRegistry(model_type)

    # 버전 목록 조회 모드
    if len(sys.argv) >= 3 and sys.argv[2] == "--list":
        versions = registry.list_versions()
        current = registry.get_current_version()
        print(f"\n=== {model_type} 버전 목록 ===")
        print(f"현재 운영 버전: {current}")
        print(f"저장된 버전 ({len(versions)}개):")
        for i, v in enumerate(versions):
            marker = " ← CURRENT" if v.startswith(current + "_") else ""
            print(f"  [{i}] {v}{marker}")
        return

    # 롤백 모드
    print(f"\n=== {model_type} 롤백 시작 ===")
    versions = registry.list_versions()
    print(f"저장된 버전: {len(versions)}개")
    print(f"롤백 전 운영 버전: {registry.get_current_version()}")

    try:
        if len(sys.argv) >= 3:
            target_version = sys.argv[2]  # 예: "v2"
            matching = [v for v in versions if v.startswith(target_version + "_")]
            if not matching:
                print(f"[ERROR] {target_version}에 해당하는 파일을 찾을 수 없습니다.")
                print(f"사용 가능한 버전: {versions}")
                sys.exit(1)
            result = registry.rollback_to(matching[0])
        else:
            # 직전 버전으로 자동 롤백
            result = registry.rollback_to_previous()

        print(f"\n[SUCCESS] 롤백 완료")
        print(f"  이전 버전: {result.get('rolled_back_from')}")
        print(f"  현재 버전: {result.get('version')}")
        print(f"  롤백 시각: {result.get('rolled_back_at')}")
        print(f"  교체된 파일: {result.get('rolled_back_to_file')}")

    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()