from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "runs" / "detect"
    / "ramp_test" / "weights" / "best.pt"
)

IMAGE_PATH = ROOT / "test01_pic.jpg"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"학습된 모델이 없습니다: {MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"테스트 이미지가 없습니다: {IMAGE_PATH}"
    )

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    save=True,
    project=str(ROOT / "results"),
    name="detection_test",
)

for result in results:
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            print(
                f"탐지 객체: {class_name}, "
                f"신뢰도: {confidence:.2f}"
            )

print("탐지 테스트 완료")
