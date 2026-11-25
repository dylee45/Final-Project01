"""경사로 통행 조건 판단 모듈."""

from math import isclose, isfinite


RAMP_LABELS = {
    "ramp", "wheelchair ramp", "경사로"
}

STAIR_LABELS = {
    "stair", "stairs", "staircase",
    "step", "steps", "계단"
}

OBSTACLE_LABELS = {
    "obstacle", "장애물"
}

# 장애물 겹침 비율 임시 판단 기준
OVERLAP_THRESHOLD = 0.30

# 최종 통행 판단 표시
VERDICT_LABELS = {
    "V": "통행 가능 후보",
    "X": "통행 조건 미충족",
    "?": "판단 보류"
}


def normalize_label(detection):
    """객체 클래스 이름을 정규화한다."""
    return (
        detection["class_name"]
        .strip()
        .casefold()
        .replace("_", " ")
        .replace("-", " ")
    )


def check_model_classes(model):
    """YOLO 모델이 지원하는 객체 클래스를 확인한다."""

    names = getattr(model, "names", {})

    if isinstance(names, dict):
        names = names.values()

    labels = {
        normalize_label({"class_name": name})
        for name in names
    }

    return {
        "ramp_supported": bool(labels & RAMP_LABELS),
        "stairs_supported": bool(labels & STAIR_LABELS),
        "obstacle_supported": bool(labels & OBSTACLE_LABELS),
        "classes": sorted(labels)
    }


def calculate_overlap(ramp_bbox, object_bbox):
    """장애물 영역 중 경사로와 겹치는 면적 비율을 계산한다."""

    left = max(ramp_bbox[0], object_bbox[0])
    top = max(ramp_bbox[1], object_bbox[1])
    right = min(ramp_bbox[2], object_bbox[2])
    bottom = min(ramp_bbox[3], object_bbox[3])

    intersection = (
        max(0, right - left)
        * max(0, bottom - top)
    )

    object_area = (
        max(0, object_bbox[2] - object_bbox[0])
        * max(0, object_bbox[3] - object_bbox[1])
    )

    if object_area <= 0:
        return 0.0

    return intersection / object_area


def check_obstacles(ramps, detections):
    """경사로와 계단 및 장애물의 위치 관계를 분석한다."""

    obstacles = [
        (index, obj)
        for index, obj in enumerate(detections)
        if normalize_label(obj)
        in STAIR_LABELS | OBSTACLE_LABELS
    ]

    conflicts = []

    for ramp_index, ramp in enumerate(ramps, start=1):
        for object_index, obstacle in obstacles:

            overlap = calculate_overlap(
                ramp["bbox"],
                obstacle["bbox"]
            )

            if overlap >= OVERLAP_THRESHOLD:
                relation = "크게 겹침"
            elif overlap > 0:
                relation = "일부 겹침"
            else:
                relation = "겹치지 않음"

            conflicts.append({
                "ramp_index": ramp_index,
                "object_index": object_index,
                "class_name": obstacle["class_name"],
                "confidence": obstacle["confidence"],
                "bbox": obstacle["bbox"],
                "overlap": round(overlap, 4),
                "relation": relation
            })

    return conflicts


def summarize_detections(detections, conflicts):
    """탐지 객체와 겹침 분석 결과를 집계한다."""

    ramps = []
    stairs = []
    obstacles = []

    for index, obj in enumerate(detections):
        label = normalize_label(obj)

        if label in RAMP_LABELS:
            ramps.append(index)

        elif label in STAIR_LABELS:
            stairs.append(index)

        elif label in OBSTACLE_LABELS:
            obstacles.append(index)

    overlapping_objects = {
        item["object_index"]
        for item in conflicts
        if item["relation"] != "겹치지 않음"
    }

    checked_objects = {
        item["object_index"]
        for item in conflicts
    }

    return {
        "ramp_count": len(ramps),
        "stairs_count": len(stairs),
        "obstacle_count": len(obstacles),
        "overlapping_count": len(overlapping_objects),
        "outside_count": len(
            checked_objects - overlapping_objects
        )
    }


def summarize_obstacles(conflicts):
    """객체별 최대 겹침 비율과 위치 관계를 반환한다."""

    objects = {}

    priority = {
        "겹치지 않음": 0,
        "일부 겹침": 1,
        "크게 겹침": 2
    }

    for item in conflicts:

        index = item["object_index"]
        previous = objects.get(index)

        if previous is None:
            objects[index] = item.copy()

        elif (
            priority[item["relation"]],
            item["overlap"]
        ) > (
            priority[previous["relation"]],
            previous["overlap"]
        ):
            objects[index] = item.copy()

    return list(objects.values())


def evaluate_accessibility(
    detections,
    measurements,
    slope_limit=None,
    retried=False,
    model_classes=None,
):
    """객체 탐지 결과와 경사각을 이용해 통행 조건을 판단한다."""

    ramps = [
        obj for obj in detections
        if normalize_label(obj) in RAMP_LABELS
    ]

    conflicts = check_obstacles(ramps, detections)

    summary = summarize_detections(
        detections,
        conflicts
    )

    obstacle_details = summarize_obstacles(conflicts)

    # 기본값: 판단 보류
    verdict = "?"

    if not ramps:

        stair_detected = any(
            normalize_label(obj) in STAIR_LABELS
            for obj in detections
        )

        if stair_detected:
            if stair_detected:
                verdict = "X"
                status = "통행 기준 미충족"
                reason = (
                    "계단이 탐지됐지만 경사로는 탐지되지 않아 "
                    "사진 속 통행 경로가 프로젝트 기준을 충족하지 않습니다."
                )

        elif retried:
            status = "확인 필요"
            reason = "재검사 후에도 경사로가 탐지되지 않았습니다."

        else:
            status = "확인 필요"
            reason = "경사로가 탐지되지 않았습니다."

    elif any(
        item["relation"] == "크게 겹침"
        for item in conflicts
    ):
        verdict = "X"
        status = "장애물 확인 필요"
        reason = "경사로와 계단 또는 장애물의 탐지 영역이 크게 겹칩니다."

    elif any(
        item["relation"] == "일부 겹침"
        for item in conflicts
    ):
        verdict = "?"
        status = "장애물 확인 필요"
        reason = "경사로와 계단 또는 장애물의 탐지 영역이 일부 겹칩니다."

    elif (
        len(measurements) != len(ramps)
        or any(
            item.get("angle") is None
            or not isfinite(item["angle"])
            or item["angle"] < 0
            for item in measurements
        )
    ):
        status = "확인 필요"
        reason = "일부 또는 전체 경사로의 각도를 추정할 수 없습니다."

    elif any(ramp.get("from_crop") for ramp in ramps):
        status = "확인 필요"
        reason = "재검사로 발견된 경사로의 추정 결과를 확인해 주세요."

    elif (
        slope_limit is None
        or not isfinite(slope_limit)
        or slope_limit <= 0
    ):
        status = "확인 필요"
        reason = "유효한 경사 기준이 설정되지 않았습니다."

    else:

        angles = [
            item["angle"]
            for item in measurements
        ]

        max_angle = max(angles)

        if (
            max_angle > slope_limit
            and not isclose(
                max_angle,
                slope_limit,
                abs_tol=1e-8
            )
        ):
            verdict = "X"
            status = "경사 기준 초과"
            reason = (
                f"최대 추정 경사각 {max_angle:.2f}도가 "
                f"설정 기준 {slope_limit:.2f}도를 초과합니다."
            )

        elif (
            model_classes is None
            or not model_classes.get("obstacle_supported")
        ):
            verdict = "?"
            status = "확인 필요"
            reason = (
                "추정 경사각은 기준 이내이지만 "
                "모델의 장애물 탐지 지원 여부가 확인되지 않았거나 "
                "장애물 탐지를 지원하지 않습니다."
            )

        else:
            verdict = "V"
            status = "경사 기준 이내"
            reason = (
                "추정 경사각이 기준 이내이며 "
                "경사로와 겹치는 장애물 후보가 탐지되지 않았습니다. "
                "실제 통행 조건은 현장에서 확인해 주세요."
            )

    warnings = []

    if model_classes is None:
        warnings.append(
            "YOLO 모델의 지원 클래스 정보가 전달되지 않았습니다."
        )

    elif not model_classes.get("obstacle_supported"):
        warnings.append(
            "현재 YOLO 모델은 장애물 클래스를 지원하지 않습니다."
        )

    return {
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[verdict],
        "status": status,
        "reason": reason,
        "conflicts": conflicts,
        "summary": summary,
        "obstacle_details": obstacle_details,
        "model_classes": model_classes,
        "warnings": warnings
    }
