from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "datasets" / "data.yaml"

model = YOLO("yolo11n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=5,
    imgsz=640,
    batch=8,
    project=str(ROOT / "runs" / "detect"),
    name="ramp_test",
)
