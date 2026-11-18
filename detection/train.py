import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    # 학습 옵션 설정
    parser = argparse.ArgumentParser(
        description="YOLO 경사로 Detection 모델 학습"
    )

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", type=str, default="ramp_detection")

    args = parser.parse_args()

    # 데이터셋 경로 확인
    data_path = Path(args.data).resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"데이터셋 설정 파일 없음: {data_path}")

    # GPU 자동 선택
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"사용 장치: {device}")
    print(f"데이터셋: {data_path}")
    print(f"학습 횟수: {args.epochs}")

    # YOLO 모델 불러오기
    model = YOLO(args.model)

    # 모델 학습
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(Path("runs/detect").resolve()),
        name=args.name
    )

    print(f"학습 결과 저장 위치: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()