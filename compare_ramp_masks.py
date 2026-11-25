"""Report where each ramp segmentation model stops in the hybrid pipeline."""

import argparse
from math import ceil, floor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

from src import analysis
from src.detection import detect_ramp_with_image, load_model


def report_masks(objects, size, bbox):
    width, height = size
    left, top = max(0, floor(bbox[0])), max(0, floor(bbox[1]))
    right, bottom = min(width, ceil(bbox[2])), min(height, ceil(bbox[3]))
    ramps = [obj for obj in objects if analysis._label(obj) in analysis.RAMP_LABELS]
    if not ramps:
        print("  NO_RAMP_PREDICTION (confidence threshold: 0.25)")
        return
    for index, ramp in enumerate(ramps, 1):
        mask = ramp.get("mask")
        prefix = f"  Candidate {index}, confidence={ramp['confidence']:.3f}"
        if mask is None or mask.shape != (height, width):
            print(f"{prefix}: MISSING_OR_MISALIGNED_MASK")
            continue
        total = np.count_nonzero(mask)
        inside = np.count_nonzero(mask[top:bottom, left:right]) if right > left and bottom > top else 0
        fraction = inside / total if total else 0
        status = "PASS" if inside >= max(60, 0.5 * total) else "REJECTED_BY_BOX_MATCH"
        print(f"{prefix}: {status}, inside={inside}/{total} ({fraction:.1%})")

def compare_model(image, model, boxes, conf):
    full, _ = detect_ramp_with_image(image, model=model, conf=conf)
    for index, ramp in enumerate(boxes, 1):
        bbox = ramp["bbox"]
        print(f"Detection #{index} ({ramp['confidence']:.1%}) | full frame")
        report_masks(full, image.size, bbox)
        if analysis._matching_ramp(full, image.size, bbox) is not None:
            print("  Result: usable full-frame mask")
            continue

        # Observe the actual crop and box passed by the app's fallback.
        def report_crop(objects, size, crop_bbox):
            print("  Crop retry:")
            report_masks(objects, size, crop_bbox)
            return original_match(objects, size, crop_bbox)

        original_match = analysis._matching_ramp
        def predict_crop(crop, model):
            return detect_ramp_with_image(crop, model=model, conf=conf)

        with patch.object(analysis, "detect_ramp_with_image", side_effect=predict_crop):
            with patch.object(analysis, "_matching_ramp", side_effect=report_crop):
                cropped = analysis._ramp_mask_in_box(image, bbox, model)
        print("  Result: usable crop mask" if cropped is not None else "  Result: no usable mask")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path, help="Path to the original photo")
    parser.add_argument("--detector", default="runs/detect/ramp_final/weights/best.pt")
    parser.add_argument("--seg3", default="runs/segment/ramp_mask/weights/best.pt")
    parser.add_argument("--seg20", default="runs/segment/ramp_mask_e20/weights/best.pt")
    args = parser.parse_args()
    with Image.open(args.photo) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((1024, 1024))

    detector = load_model(analysis.resolve_weights(args.detector))
    objects, _ = detect_ramp_with_image(image, model=detector)
    obstacles = [obj for obj in objects if analysis._label(obj) in analysis.OBSTACLE_LABELS]
    boxes = [
        obj for obj in objects if analysis._label(obj) in analysis.RAMP_LABELS
        and not any(analysis._boxes_conflict(obj["bbox"], obstacle["bbox"]) for obstacle in obstacles)
    ]
    if not boxes:
        print("No ramp box eligible for segmentation. Check Detection and obstacle overlap.")
        return

    for label, path in (("3 epochs", args.seg3), ("20 epochs", args.seg20)):
        print(f"\n=== {label} | {path} ===")
        model = load_model(analysis.resolve_weights(path))
        if getattr(model, "task", None) != "segment":
            parser.error(f"Not a segmentation model: {path}")
        conf = 0.10 if label == "20 epochs" else 0.25
        compare_model(image, model, boxes, conf)


if __name__ == "__main__":
    main()
