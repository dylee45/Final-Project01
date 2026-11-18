from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "data.yaml"

PROJECT_DIR = ROOT / "runs" / "detect"
RUN_NAME = "ramp_test_v2"


def main():

    # 데이터셋 설정 파일 확인
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"데이터셋 설정 파일이 없습니다: {DATA_YAML}"
        )

    # 기존 학습 결과 보호
    output_dir = PROJECT_DIR / RUN_NAME

    if output_dir.exists():
        raise FileExistsError(
            f"이미 학습 결과 폴더가 존재합니다: {output_dir}\n"
            "기존 결과를 보존하려면 RUN_NAME을 변경하세요."
        )

    print("YOLO11n 재학습 시작")
    print(f"데이터셋: {DATA_YAML}")
    print("Epoch: 20")
    print("Batch: 4")
    print("Device: CPU")

    # 사전학습된 YOLO11n 모델 불러오기
    model = YOLO(str(ROOT / "yolo11n.pt"))

    # 모델 학습
    model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,
        plots=True
    )

    print("\nYOLO 재학습 완료!")
    print(f"학습 결과: {output_dir}")
    print(f"모델 파일: {output_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
