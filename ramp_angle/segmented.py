"""Fit a ramp plane and its adjacent reference plane using segmentation masks."""

import numpy as np

from ramp_angle.automatic import AngleUnavailable, _angle_between, _estimate_across_edge, _fit_samples


def _fit_mask(depth, mask, focal, label):
    import cv2

    # Leave a margin so uncertain pixels at mask boundaries never enter the fit.
    interior = cv2.erode(
        mask.astype(np.uint8), np.ones((5, 5), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )
    rows, cols = np.nonzero(interior)
    if len(rows) < 60 or min(np.ptp(rows), np.ptp(cols)) < 12:
        raise AngleUnavailable(f"{label} 마스크 내부의 표본 영역이 너무 작습니다.")
    if len(rows) > 1024:
        indices = np.random.default_rng(42).choice(len(rows), size=1024, replace=False)
        rows, cols = rows[indices], cols[indices]
    normal, _, pixels = _fit_samples(depth, cols, rows, rows, (), focal, label)
    return normal, pixels


def estimate_segmented_angle(
    depth, ramp_mask, ground_masks, excluded_masks, fov, *, image=None, bbox=None,
    spread_samples=False,
):
    import cv2

    if not 0 < fov < 180:
        raise ValueError("가로 화각은 0°보다 크고 180°보다 작아야 합니다.")
    if ramp_mask is None:
        raise AngleUnavailable("경사로 마스크를 얻지 못했습니다.")
    if not ground_masks:
        raise AngleUnavailable("기준 바닥 마스크가 탐지되지 않았습니다.")
    if any(mask.shape != depth.shape for mask in (ramp_mask, *ground_masks, *excluded_masks)):
        raise AngleUnavailable("마스크의 좌표가 분석 사진과 일치하지 않습니다.")

    excluded = np.zeros(depth.shape, dtype=bool)
    for mask in excluded_masks:
        excluded |= mask
    ramp_core = ramp_mask & ~excluded
    ramp_rows, ramp_cols = np.nonzero(ramp_core)
    if len(ramp_rows) < 60:
        raise AngleUnavailable("경사로 마스크의 유효 영역이 부족합니다.")

    # A small gap is allowed because predicted masks rarely meet exactly.
    gap = int(np.clip(0.06 * min(np.ptp(ramp_rows), np.ptp(ramp_cols)), 5, 24))
    distance = cv2.distanceTransform((~ramp_core).astype(np.uint8), cv2.DIST_L2, 5)
    nearby_ground = []
    for ground_mask in ground_masks:
        ground_core = ground_mask & ~ramp_mask & ~excluded
        if np.count_nonzero(ground_core & (distance <= gap)) >= 20:
            nearby_ground.append((ground_mask, ground_core))
    height, width = depth.shape
    focal = width / (2 * np.tan(np.radians(fov / 2)))
    combined_ground = np.logical_or.reduce(ground_masks) & ~excluded
    masks_overlap = np.count_nonzero(ramp_core & combined_ground) >= 0.15 * len(ramp_rows)
    if len(nearby_ground) != 1 or masks_overlap:
        if image is not None and bbox is not None:
            clipped_box = np.clip(bbox, 0, [width, height, width, height])
            result = _estimate_across_edge(
                image, depth, clipped_box, (), focal,
                surface_masks=(ramp_core, combined_ground),
                spread_samples=spread_samples,
            )
            if result is not None:
                return result
        raise AngleUnavailable("경사로와 기준 바닥 마스크를 입구 경계와 깊이로 구분하지 못했습니다.")

    ground_mask, ground_core = nearby_ground[0]
    ramp_core &= ~ground_mask
    ramp_normal, ramp_points = _fit_mask(depth, ramp_core, focal, "경사로")
    ground_normal, ground_points = _fit_mask(depth, ground_core, focal, "기준 바닥")
    return _angle_between(ramp_normal, ground_normal), ramp_points, ground_points
