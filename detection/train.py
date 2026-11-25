import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    # 학습 옵션 설정
    parser = argparse.ArgumentParser(
        description="YOLO 경사로 탐지·세그멘테이션 모델 학습"
    )

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--task", choices=("detect", "segment"), default="detect")
    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", type=str)

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
    default_model = "yolo11n-seg.pt" if args.task == "segment" else "yolo11n.pt"
    model = YOLO(args.model or default_model)
    task = model.task
    if args.task == "segment" and task != "segment":
        parser.error("--task segment에는 세그멘테이션 모델이 필요합니다.")

    # 모델 학습
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str((Path("runs") / task).resolve()),
        name=args.name or ("ramp_seg" if task == "segment" else "ramp_detection")
    )

    print(f"학습 결과 저장 위치: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()