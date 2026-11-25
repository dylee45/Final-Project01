"""Run the app's hybrid angle analysis with an adjustable ramp-mask threshold."""

import argparse
from pathlib import Path

from PIL import Image

from src.analysis import analyze_image, resolve_weights
from src.detection import RampSegmentation, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--detector", default="runs/detect/ramp_final/weights/best.pt")
    parser.add_argument("--ground", default="runs/segment/ramp_seg/weights/best.pt")
    parser.add_argument("--ramp", default="runs/segment/ramp_mask_e20/weights/best.pt")
    parser.add_argument("--ramp-conf", type=float, default=0.20)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--spread-samples", action="store_true", help="Use a wider sample strip at the ramp entrance")
    parser.add_argument("--preview", type=Path, help="Save the annotated image to this path")
    args = parser.parse_args()
    if not 0 < args.ramp_conf <= 1:
        parser.error("--ramp-conf must be greater than 0 and at most 1")
    if not 0 < args.fov < 180:
        parser.error("--fov must be between 0 and 180 degrees")

    print(f"Ramp segmentation: {args.ramp} | threshold: {args.ramp_conf:.0%}", flush=True)
    if args.spread_samples:
        print("입구 경계 추정 시 표본 범위 확장", flush=True)
    with Image.open(args.photo) as image:
        detector = load_model(resolve_weights(args.detector))
        ground_model = load_model(resolve_weights(args.ground))
        ramp_model = RampSegmentation(load_model(resolve_weights(args.ramp)), args.ramp_conf)
        result = analyze_image(
            image, detector, args.fov, ground_model=ground_model, ramp_segment_model=ramp_model,
            spread_samples=args.spread_samples,
        )

    for index, measurement in enumerate(result["ramps"], 1):
        angle = measurement["angle"]
        if angle is None:
            print(f"경사로 {index} 추정 불가 · {measurement['reason']}")
        else:
            print(f"경사로 {index} 추정 경사각 · {angle:.2f}°")
    if not result["ramps"]:
        print(result["reason"])
    if args.preview is not None:
        result["preview"].save(args.preview)
        print(f"결과 이미지 · {args.preview.resolve()}")


if __name__ == "__main__":
    main()
