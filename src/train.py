from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "data.yaml"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"데이터셋 설정 파일이 없습니다: {DATA_YAML}"
        )

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=5,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs" / "detect"),
        name="ramp_test",
        exist_ok=True
    )

    print("YOLO 학습 완료!")


if __name__ == "__main__":
    main()
