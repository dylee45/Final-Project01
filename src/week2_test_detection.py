import csv
from pathlib import Path

from detection import load_model, detect_ramp_with_image


ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = ROOT / "test_images"
OUTPUT_DIR = ROOT / "results" / "batch_test_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = load_model(
    ROOT / "runs" / "detect"
    / "ramp_test_v2" / "weights" / "best.pt"
)

image_files = sorted(
    file
    for file in TEST_DIR.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

records = []

for image_path in image_files:

    print(f"\n테스트 중: {image_path.name}")

    detections, annotated_image = detect_ramp_with_image(
        str(image_path),
        model=model
    )

    annotated_image.save(
        OUTPUT_DIR / image_path.name
    )

    max_confidence = max(
        (d["confidence"] for d in detections),
        default=0.0
    )

    records.append({
        "filename": image_path.name,
        "detection_count": len(detections),
        "ramp_detected": bool(detections),
        "max_confidence": round(max_confidence, 4)
    })

    print(f"탐지 개수: {len(detections)}")
    print(f"최대 신뢰도: {max_confidence:.2f}")

csv_path = OUTPUT_DIR / "test_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "detection_count",
            "ramp_detected",
            "max_confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

print("\n20장 테스트 완료!")
print(f"결과 저장 위치: {OUTPUT_DIR}")
