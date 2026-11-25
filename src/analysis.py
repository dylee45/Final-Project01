"""Connect YOLO detection with automatic depth-based ramp analysis."""

from src.accessibility import check_model_classes, evaluate_accessibility
from math import ceil, floor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ramp_angle.automatic import AngleUnavailable, estimate_ramp_angle
from ramp_angle.estimate import estimate_depth
from ramp_angle.segmented import estimate_segmented_angle
from src.detection import ROOT, detect_ramp_with_image

RAMP_LABELS = {"ramp", "wheelchair ramp", "경사로"}
GROUND_LABELS = {"level ground", "ground", "기준 바닥", "기준면"}
OBSTACLE_LABELS = {"stair", "stairs", "staircase", "step", "steps", "계단", "obstacle", "장애물"}

DEFAULT_WEIGHTS = (
    ROOT / "best.pt",
    ROOT / "runs/detect/ramp_final/weights/best.pt",
    ROOT / "runs/detect/ramp_test_v2/weights/best.pt",
    ROOT / "runs/detect/ramp_test/weights/best.pt",
)


def resolve_weights(value):
    if not value.strip():
        for path in DEFAULT_WEIGHTS:
            if path.is_file():
                return path
        raise FileNotFoundError("프로젝트 폴더에 학습한 best.pt를 넣거나 모델 경로를 입력해 주세요.")
    path = Path(value.strip().strip('"')).expanduser()
    path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
    return path


def _label(detection):
    return detection["class_name"].strip().casefold().replace("_", " ").replace("-", " ")


def _retry_ramp_crop(image, model):
    # Increase the fraction of the input occupied by an entrance ramp.
    # Ground prediction and depth estimation continue to use the full image.
    top = round(image.height * 0.40)
    crop = image.crop((0, top, image.width, image.height))
    results, _ = detect_ramp_with_image(crop, model=model)
    ramps = []
    for obj in results:
        mask = obj.get("mask")
        if _label(obj) not in RAMP_LABELS or mask is None:
            continue
        if mask.shape != (crop.height, crop.width) or not mask.any():
            continue
        left, upper, right, lower = obj["bbox"]
        ramps.append({
            **obj,
            "bbox": [left, upper + top, right, lower + top],
            "mask": np.pad(mask, ((top, 0), (0, 0)), constant_values=False),
            "from_crop": True,
        })
    return ramps


def _dual_segmentation(image, ramp_model, ground_model):
    if any(getattr(model, "task", None) != "segment" for model in (ramp_model, ground_model)):
        raise ValueError("모델을 두 개 사용할 때는 경사로와 바닥 모델 모두 세그멘테이션 모델이어야 합니다.")

    ramp_results, _ = detect_ramp_with_image(image, model=ramp_model)
    ground_results, _ = detect_ramp_with_image(image, model=ground_model)
    ground_objects = [
        obj for obj in ground_results if _label(obj) in GROUND_LABELS | OBSTACLE_LABELS
    ]
    ramps = [obj for obj in ramp_results if _label(obj) in RAMP_LABELS]
    retried = not ramps and image.height >= 80
    if retried:
        ramps = _retry_ramp_crop(image, ramp_model)
    detections = ground_objects + ramps
    preview = image.copy()
    _draw_masks(preview, detections)
    return detections, preview, retried


def _draw_masks(preview, objects):
    for obj in objects:
        mask = obj.get("mask")
        if mask is None or mask.shape != (preview.height, preview.width):
            continue
        label = _label(obj)
        if label in RAMP_LABELS:
            color = "deepskyblue"
        elif label in GROUND_LABELS:
            color = "darkorange"
        else:
            color = "crimson"
        tint = Image.new("RGB", preview.size, color)
        alpha = Image.fromarray(mask.astype("uint8") * 75)
        preview.paste(tint, (0, 0), alpha)


def _box_overlap(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area, second_area = ((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in (first, second))
    return intersection, first_area, second_area


def _boxes_conflict(first, second):
    intersection, first_area, second_area = _box_overlap(first, second)
    smaller_area = min(first_area, second_area)
    return smaller_area > 0 and intersection >= 0.30 * smaller_area


def _same_ramp(first, second):
    first_mask, second_mask = first.get("mask"), second.get("mask")
    if first_mask is not None and second_mask is not None:
        if first_mask is second_mask:
            return True
        intersection = np.count_nonzero(first_mask & second_mask)
        union = np.count_nonzero(first_mask | second_mask)
        return union > 0 and intersection >= 0.75 * union
    intersection, first_area, second_area = _box_overlap(first["bbox"], second["bbox"])
    union = first_area + second_area - intersection
    return union > 0 and intersection >= 0.75 * union


def _deduplicate_ramps(ramps):
    # Prefer a ramp with a usable mask. Among duplicates with equivalent mask
    # availability, keep the box with the higher detection confidence.
    ranked = sorted(ramps, key=lambda ramp: ("mask" in ramp, ramp["confidence"]), reverse=True)
    kept = []
    for ramp in ranked:
        if not any(_same_ramp(ramp, other) for other in kept):
            kept.append(ramp)
    kept_ids = {id(ramp) for ramp in kept}
    return [ramp for ramp in ramps if id(ramp) in kept_ids]


def _draw_boxes(preview, detections):
    draw = ImageDraw.Draw(preview)
    for obj in detections:
        x0, y0, x1, y1 = obj["bbox"]
        color = "blue" if _label(obj) in RAMP_LABELS else "crimson"
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f"{obj['class_name']} {obj['confidence']:.2f}"
        origin = (x0, max(0, y0 - 17))
        draw.rectangle(draw.textbbox(origin, label), fill=color)
        draw.text(origin, label, fill="white")


def _matching_ramp(objects, image_size, bbox):
    width, height = image_size
    left, top = max(0, floor(bbox[0])), max(0, floor(bbox[1]))
    right, bottom = min(width, ceil(bbox[2])), min(height, ceil(bbox[3]))
    if right <= left or bottom <= top:
        return None

    best, best_overlap = None, 0
    for obj in objects:
        mask = obj.get("mask")
        if _label(obj) not in RAMP_LABELS:
            continue
        if mask is None or mask.shape != (height, width):
            continue
        overlap = np.count_nonzero(mask[top:bottom, left:right])
        if overlap >= max(60, 0.5 * np.count_nonzero(mask)) and overlap > best_overlap:
            best, best_overlap = obj, overlap
    return best


def _ramp_mask_in_box(image, bbox, model):
    left, top = max(0, floor(bbox[0])), max(0, floor(bbox[1]))
    right, bottom = min(image.width, ceil(bbox[2])), min(image.height, ceil(bbox[3]))
    width, height = right - left, bottom - top
    if min(width, height) < 12:
        return None

    pad_x, pad_y = ceil(width * 0.15), ceil(height * 0.15)
    x0, y0 = max(0, left - pad_x), max(0, top - pad_y)
    x1, y1 = min(image.width, right + pad_x), min(image.height, bottom + pad_y)
    crop = image.crop((x0, y0, x1, y1))
    objects, _ = detect_ramp_with_image(crop, model=model)
    match = _matching_ramp(objects, crop.size, (left - x0, top - y0, right - x0, bottom - y0))
    if match is None:
        return None

    full_mask = np.zeros((image.height, image.width), dtype=bool)
    full_mask[y0:y1, x0:x1] = match["mask"]
    a, b, c, d = match["bbox"]
    return full_mask, [a + x0, b + y0, c + x0, d + y0]


def _detect_then_segment(image, detector, ramp_model, ground_model):
    if getattr(detector, "task", None) != "detect":
        raise ValueError("첫 번째 경로에 경사로·계단 Detection 모델을 입력해 주세요.")
    if any(getattr(model, "task", None) != "segment" for model in (ramp_model, ground_model)):
        raise ValueError("경사로 분할과 기준 바닥에는 각각 Segmentation 모델을 입력해 주세요.")

    detections, preview = detect_ramp_with_image(image, model=detector)
    obstacles = [obj for obj in detections if _label(obj) in OBSTACLE_LABELS]
    ramps = [obj for obj in detections if _label(obj) in RAMP_LABELS]
    full_frame_results = None
    for ramp in ramps:
        if any(_boxes_conflict(ramp["bbox"], obstacle["bbox"]) for obstacle in obstacles):
            ramp["mask_error"] = "경사로 후보가 계단 또는 장애물 탐지 영역과 크게 겹칩니다."
            continue
        if full_frame_results is None:
            full_frame_results, _ = detect_ramp_with_image(image, model=ramp_model)
        match = _matching_ramp(full_frame_results, image.size, ramp["bbox"])
        if match is not None:
            ramp["mask"] = match["mask"]
            ramp["angle_bbox"] = match["bbox"]
            continue

        cropped = _ramp_mask_in_box(image, ramp["bbox"], ramp_model)
        if cropped is None:
            ramp["mask_error"] = "원본과 탐지 박스 주변에서 경사면 마스크를 얻지 못했습니다."
        else:
            ramp["mask"], ramp["angle_bbox"] = cropped

    filtered_ramps = _deduplicate_ramps(ramps)
    duplicates_removed = len(filtered_ramps) != len(ramps)
    if duplicates_removed:
        kept_ids = {id(ramp) for ramp in filtered_ramps}
        detections = [
            obj for obj in detections if _label(obj) not in RAMP_LABELS or id(obj) in kept_ids
        ]
        preview = image.copy()
    if any("mask" in ramp for ramp in filtered_ramps):
        ground_results, _ = detect_ramp_with_image(image, model=ground_model)
        grounds = [obj for obj in ground_results if _label(obj) in GROUND_LABELS]
        detections.extend(grounds)
        _draw_masks(preview, grounds + filtered_ramps)
    if duplicates_removed:
        _draw_boxes(preview, [obj for obj in detections if _label(obj) not in GROUND_LABELS])
    return detections, preview


def _draw_obstacle_relations(preview, details):
    """Draw image-space overlap, not physical obstruction."""
    draw = ImageDraw.Draw(preview)
    colors = {"크게 겹침": "red", "일부 겹침": "darkorange", "겹치지 않음": "green"}
    for item in details:
        x0, y0, x1, y1 = item["bbox"]
        color = colors[item["relation"]]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f"Overlap {item['overlap']:.1%}"
        origin = (max(0, x0), max(0, y0 - 32))
        draw.rectangle(draw.textbbox(origin, label), fill=color)
        draw.text(origin, label, fill="white")


def analyze_image(
    image, model, fov=60.0, slope_limit=None, *, ground_model=None, ramp_segment_model=None,
    spread_samples=False,
):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1024, 1024))
    retried = False
    hybrid = ramp_segment_model is not None
    if hybrid:
        detections, preview = _detect_then_segment(image, model, ramp_segment_model, ground_model)
    elif ground_model is None:
        detections, preview = detect_ramp_with_image(image, model=model)
    else:
        detections, preview, retried = _dual_segmentation(image, model, ground_model)
    ramps = [obj for obj in detections if _label(obj) in RAMP_LABELS]
    measurements = []
    segmented = hybrid or getattr(model, "task", None) == "segment"
    ground_masks = [
        obj["mask"] for obj in detections if _label(obj) in GROUND_LABELS and "mask" in obj
    ]
    foreground = [
        obj for obj in detections if _label(obj) not in GROUND_LABELS and "mask" in obj
    ]
    if ramps:
        depth = None
        draw = ImageDraw.Draw(preview)
        for ramp in ramps:
            try:
                if hybrid:
                    if "mask_error" in ramp:
                        raise AngleUnavailable(ramp["mask_error"])
                    if not ground_masks:
                        raise AngleUnavailable("원본 사진에서 기준 바닥 마스크를 얻지 못했습니다.")
                if depth is None:
                    depth = estimate_depth(image)
                    if depth.shape != (image.height, image.width):
                        raise ValueError("깊이 결과의 크기가 분석 사진과 일치하지 않습니다.")
                if segmented:
                    excluded_masks = [obj["mask"] for obj in foreground if obj is not ramp]
                    angle, ramp_points, ground_points = estimate_segmented_angle(
                        depth, ramp.get("mask"), ground_masks, excluded_masks, fov,
                        image=image, bbox=ramp.get("angle_bbox", ramp["bbox"]),
                        spread_samples=spread_samples,
                    )
                else:
                    other_boxes = [obj["bbox"] for obj in detections if obj is not ramp]
                    angle, ramp_points, ground_points = estimate_ramp_angle(
                        depth, ramp["bbox"], other_boxes, fov, image=image
                    )
            except AngleUnavailable as error:
                measurements.append({"angle": None, "reason": str(error)})
                continue
            measurements.append({"angle": angle, "reason": None})
            for pixels, color in ((ramp_points, "cyan"), (ground_points, "orange")):
                for x, y in pixels.tolist():
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    # 통행 조건 판단 모듈 호출
    model_classes = check_model_classes(model)
    if not hybrid and ground_model is not None:
        ground_classes = check_model_classes(ground_model)
        model_classes["classes"] = sorted(set(model_classes["classes"]) | set(ground_classes["classes"]))
        for key in ("stairs_supported", "obstacle_supported"):
            model_classes[key] = ground_classes[key]

    accessibility_result = evaluate_accessibility(
        detections=detections,
        measurements=measurements,
        slope_limit=slope_limit,
        retried=retried,
        model_classes=model_classes,
    )
    _draw_obstacle_relations(preview, accessibility_result["obstacle_details"])

    return {
        "verdict": accessibility_result["verdict"],
        "verdict_label": accessibility_result["verdict_label"],
        "status": accessibility_result["status"],
        "reason": accessibility_result["reason"],
        "conflicts": accessibility_result["conflicts"],
        "summary": accessibility_result["summary"],
        "obstacle_details": accessibility_result["obstacle_details"],
        "model_classes": accessibility_result["model_classes"],
        "warnings": accessibility_result["warnings"],
        "detections": [
            {k: v for k, v in obj.items() if k != "mask"}
            for obj in detections
        ],
        "ramps": measurements,
        "preview": preview,
    }