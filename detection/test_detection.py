from ultralytics import YOLO


def main():
    # 사전 학습된 YOLO11 Object Detection 모델 불러오기
    model = YOLO("yolo11n.pt")

    # GPU 사용 가능 여부 확인
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"사용 장치: {device}")

    # 샘플 이미지로 Detection 테스트
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        device=device,
        conf=0.25,
        save=False,
    )

    # 탐지 결과 출력
    for result in results:
        print(f"탐지된 객체 수: {len(result.boxes)}")

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            print(f"{class_name}: {confidence:.2f}")


if __name__ == "__main__":
    main()