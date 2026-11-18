import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 경사로 객체 탐지"
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", type=str, default="ramp_predict")

    args = parser.parse_args()

    # 모델 파일 확인
    model_path = Path(args.model).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"모델 파일 없음: {model_path}"
        )

    # GPU 자동 선택
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(model_path))

    # 객체 탐지
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        save=True,
        project=str(Path("runs/predict").resolve()),
        name=args.name
    )

    # 탐지 결과 출력
    for result in results:
        print(f"\n이미지: {result.path}")
        print(f"탐지 객체 수: {len(result.boxes)}")

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            print(f"{class_name}: {confidence:.2f}")

    print(f"결과 저장 위치: {results[0].save_dir}")


if __name__ == "__main__":
    main()