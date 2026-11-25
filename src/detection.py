from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "runs" / "detect"
    / "ramp_test" / "weights" / "best.pt"
)

IMAGE_PATH = ROOT / "test01_pic.jpg"


class RampSegmentation:
    """Set the prediction threshold for the ramp mask model only."""

    def __init__(self, model, confidence):
        self.model = model
        self.task = model.task
        self.confidence = confidence

    def predict(self, *, conf, **kwargs):
        return self.model.predict(conf=self.confidence, **kwargs)


def load_model(model_path=None):
    if model_path is None:
        model_path = MODEL_PATH

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {model_path}"
        )

    from ultralytics import YOLO

    return YOLO(str(model_path))


def detect_ramp(image, model=None, conf=0.25):
    """
    사진을 입력받아 YOLO 탐지 결과 반환

    image:
        이미지 파일 경로 또는 PIL 이미지

    return:
        탐지 객체 정보 리스트
    """

    if model is None:
        model = load_model()

    results = model.predict(
        source=image,
        conf=conf,
        verbose=False
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            bbox = [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ]

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox
            })

    return detections


def detect_ramp_with_image(image, model=None, conf=0.25):
    """
    탐지 결과와 Bounding Box가 그려진 이미지 반환
    """

    if model is None:
        model = load_model()

    results = model.predict(
        source=image,
        conf=conf,
        verbose=False,
        retina_masks=getattr(model, "task", None) == "segment",
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0])

            detections.append({
                "class_name": result.names[class_id],
                "confidence": float(box.conf[0]),
                "bbox": [
                    round(float(value), 2)
                    for value in box.xyxy[0].tolist()
                ]
            })
            masks = getattr(result, "masks", None)
            if masks is not None and index < len(masks.data):
                detections[-1]["mask"] = masks.data[index].cpu().numpy() > 0.5

    # YOLO plot()은 BGR 배열을 반환하므로 RGB로 변환
    annotated_image = Image.fromarray(
        results[0].plot()[..., ::-1]
    )

    return detections, annotated_image


if __name__ == "__main__":

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"테스트 이미지가 없습니다: {IMAGE_PATH}"
        )

    print("YOLO Detection 테스트 시작")

    model = load_model()

    detections, annotated_image = detect_ramp_with_image(
        str(IMAGE_PATH),
        model=model
    )

    if detections:
        for detection in detections:
            print(
                f"탐지 객체: {detection['class_name']}, "
                f"신뢰도: {detection['confidence']:.2f}, "
                f"BBox: {detection['bbox']}"
            )
    else:
        print("탐지된 객체가 없습니다.")

    output_dir = ROOT / "results" / "detection_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "result.jpg"
    annotated_image.save(output_path)

    print(f"결과 이미지 저장: {output_path}")
    print("탐지 테스트 완료")
