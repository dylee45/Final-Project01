"""Estimate ramp inclination using depth samples inside and below a YOLO box."""

import numpy as np

from ramp_angle.estimate import fit_plane, pixel_rays


class AngleUnavailable(ValueError):
    """The image does not provide enough evidence to estimate the angle."""


def _sample_plane(depth, region, excluded_boxes, focal, label):
    height, width = depth.shape
    x0, y0, x1, y1 = np.rint(region).astype(int)
    x0, x1 = np.clip([x0, x1], 0, width)
    y0, y1 = np.clip([y0, y1], 0, height)
    if min(x1 - x0, y1 - y0) < 12:
        raise AngleUnavailable(f"{label} 영역이 사진에서 충분히 보이지 않습니다.")

    # Sample a uniform grid so the fit is not dominated by one part of the image.
    xs = np.linspace(x0, x1 - 1, min(x1 - x0, 32), dtype=int)
    ys = np.linspace(y0, y1 - 1, min(y1 - y0, 32), dtype=int)
    cols, rows = (axis.ravel() for axis in np.meshgrid(xs, ys))
    normal, _, pixels = _fit_samples(depth, cols, rows, rows, excluded_boxes, focal, label)
    return normal, pixels


def _fit_samples(depth, cols, rows, row_offsets, excluded_boxes, focal, label):
    height, width = depth.shape
    x_span = np.ptp(cols)
    y_span = np.ptp(row_offsets)
    z = depth[rows, cols]
    valid = np.isfinite(z) & (z > 0)
    for left, top, right, bottom in excluded_boxes:
        valid &= ~((cols >= left) & (cols < right) & (rows >= top) & (rows < bottom))
    if valid.sum() < max(60, len(z) * 0.5):
        raise AngleUnavailable(f"{label}의 유효한 깊이 점이 부족합니다.")

    cols, rows, row_offsets, z = cols[valid], rows[valid], row_offsets[valid], z[valid]
    points = pixel_rays(cols, rows, width, height, focal) * z[:, None]
    # RANSAC rejects part of the scene that may be included in the rectangle.
    rng = np.random.default_rng(42)
    triples = points[rng.integers(len(points), size=(96, 3))]
    normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    candidates = lengths > 1e-8
    if not candidates.any():
        raise AngleUnavailable(f"{label} 평면을 찾을 수 없습니다.")
    normals = normals[candidates] / lengths[candidates, None]
    offsets = np.sum(triples[candidates, 0] * normals, axis=1)
    distances = np.abs(points @ normals.T - offsets)
    tolerance = 0.01 * np.median(z)
    inliers = distances <= tolerance
    best = inliers[:, inliers.sum(axis=0).argmax()]

    if (
        best.sum() < max(60, 0.70 * len(points))
        or np.ptp(cols[best]) < 0.60 * x_span
        or np.ptp(row_offsets[best]) < 0.60 * y_span
    ):
        raise AngleUnavailable(f"{label}에서 일관된 표면을 찾을 수 없습니다.")

    try:
        center, normal = fit_plane(points[best])
    except (ValueError, np.linalg.LinAlgError) as error:
        raise AngleUnavailable(f"{label} 평면을 계산할 수 없습니다.") from error
    # For a floor photographed approximately upright, its normal has a
    # meaningful vertical component. Reject nearly vertical surfaces.
    if abs(normal[1]) < 0.35:
        raise AngleUnavailable(f"{label}에 바닥 대신 벽이나 옆면이 포함됐을 수 있습니다.")

    pixels = np.column_stack((cols[best], rows[best]))
    preview_indices = rng.choice(len(pixels), size=min(80, len(pixels)), replace=False)
    return normal, float(center @ normal), pixels[preview_indices]


def estimate_ramp_angle(depth, bbox, other_boxes=(), fov=60.0, *, image=None):
    """Return plane-to-plane inclination and automatically selected image points."""
    if not 0 < fov < 180:
        raise ValueError("가로 화각은 0°보다 크고 180°보다 작아야 합니다.")
    height, width = depth.shape
    x0, y0, x1, y1 = bbox
    x0, x1 = np.clip([x0, x1], 0, width)
    y0, y1 = np.clip([y0, y1], 0, height)
    box_width, box_height = x1 - x0, y1 - y0
    if min(box_width, box_height) < 24:
        raise AngleUnavailable("경사로가 사진에서 너무 작게 보입니다.")

    center_left = x0 + box_width * 0.25
    center_right = x1 - box_width * 0.25
    ramp_region = (
        center_left, y0 + box_height * 0.25,
        center_right, y1 - box_height * 0.15,
    )
    # Use only the ground in front of the ramp. Sidewalks, walls and other
    # ramps at the sides are not interchangeable reference planes.
    ground_region = (
        center_left, y1 + max(3, box_height * 0.04),
        center_right, y1 + max(24, box_height * 0.45),
    )
    focal = width / (2 * np.tan(np.radians(fov / 2)))
    try:
        ground_normal, ground_points = _sample_plane(
            depth, ground_region, other_boxes, focal, "기준 바닥"
        )
        ramp_normal, ramp_points = _sample_plane(
            depth, ramp_region, other_boxes, focal, "경사로"
        )
    except AngleUnavailable as error:
        if image is None:
            raise
        result = _estimate_across_edge(image, depth, (x0, y0, x1, y1), other_boxes, focal)
        if result is None:
            raise AngleUnavailable("경사로와 기준 바닥에 해당하는 두 표면을 구분하지 못했습니다.") from error
        return result
    return _angle_between(ramp_normal, ground_normal), ramp_points, ground_points


def _angle_between(first, second):
    cosine = np.clip(abs(first @ second), 0, 1)
    return float(np.degrees(np.arccos(cosine)))


def _entrance_edges(image, box):
    import cv2

    left, top, right, bottom = box
    width, height = right - left, bottom - top
    x0, x1 = round(left + width * 0.15), round(right - width * 0.15)
    y0, y1 = round(top + height * 0.26), round(bottom - height * 0.02)
    gray = cv2.cvtColor(np.asarray(image)[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 105)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(12, round(width * 0.10)),
        minLineLength=max(14, round(width * 0.23)), maxLineGap=max(5, round(width * 0.07)),
    )
    if segments is None:
        return
    candidates = []
    distances = np.unique(np.rint(height * np.array([0.025, 0.045, 0.065])).astype(int))
    center = (left + right) / 2
    cols = np.arange(x0, x1)
    for xa, ya, xb, yb in segments.reshape(-1, 4):
        dx = int(xb) - int(xa)
        if abs(dx) < width * 0.22:
            continue
        slope = (int(yb) - int(ya)) / dx
        intercept = y0 + ya - slope * (x0 + xa)
        at_center = slope * center + intercept
        if abs(slope) > 0.70 or not top + height * 0.34 <= at_center <= bottom - height * 0.09:
            continue
        # Count edge support across the center, including gaps between pavers.
        rows = np.rint(slope * cols + intercept).astype(int) - y0
        support = np.zeros(len(cols), dtype=bool)
        for offset in range(-3, 4):
            nearby = rows + offset
            visible = (nearby >= 0) & (nearby < len(edges))
            support[visible] |= edges[nearby[visible], cols[visible] - x0] > 0
        edge_coverage = support.mean()
        if edge_coverage < 0.38:
            continue

        # A boundary between two surfaces changes appearance on a broad band.
        # A thin grout line can have strong edges but similar surfaces on both sides.
        interior = (rows - distances[-1] >= 0) & (rows + distances[-1] < len(gray))
        if interior.mean() < 0.70:
            continue
        valid_rows = rows[interior, None]
        valid_cols = cols[interior, None] - x0
        above = gray[valid_rows - distances, valid_cols].mean(axis=1)
        below = gray[valid_rows + distances, valid_cols].mean(axis=1)
        difference = above - below
        contrast = np.median(difference)
        if abs(contrast) < 10 or np.mean(difference * np.sign(contrast) >= 6) < 0.65:
            continue
        score = abs(contrast) * np.sqrt(edge_coverage)
        candidates.append((score, slope, intercept, at_center))

    if not candidates:
        return
    ranked = sorted(candidates, reverse=True)
    best_score, best_slope, _, best_center = ranked[0]
    tried = []
    for score, slope, intercept, at_center in ranked:
        # Test refinements around the same boundary, without dropping down to
        # a different pattern on the sidewalk when the depth check fails.
        if score < 0.75 * best_score or (
            abs(at_center - best_center) > max(7, 0.045 * height)
            or abs(slope - best_slope) > 0.13
        ):
            continue
        if any(
            abs(slope - previous_slope) < 0.035 and abs(at_center - previous_center) < 5
            for previous_slope, previous_center in tried
        ):
            continue
        tried.append((slope, at_center))
        yield slope, intercept
        if len(tried) == 10:
            break


def _estimate_across_edge(
    image, depth, box, other_boxes, focal, *, surface_masks=None, spread_samples=False,
):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if width < 60 or height < 72:
        return None
    sample_cols = np.linspace(left + width * 0.27, right - width * 0.25, 32)
    cols = np.tile(sample_cols, 20)
    if spread_samples:
        # Extend farther up the ramp while keeping the reference ground local.
        ramp_offsets = np.linspace(max(5, height * 0.07), height * 0.26, 20)
        ground_offsets = np.linspace(max(5, height * 0.065), height * 0.19, 20)
    else:
        ramp_offsets = ground_offsets = np.linspace(max(5, height * 0.035), height * 0.19, 20)
    offsets_by_surface = [np.repeat(offsets, len(sample_cols)) for offsets in (ramp_offsets, ground_offsets)]
    columns = np.rint(cols).astype(int)
    image_height, image_width = depth.shape
    for slope, intercept in _entrance_edges(image, box):
        seam = slope * cols + intercept
        planes = []
        for index, (direction, label) in enumerate(((-1, "경사로"), (1, "기준 바닥"))):
            offsets = offsets_by_surface[index]
            rows = np.rint(seam + direction * offsets).astype(int)
            limit = image_height if surface_masks is not None and direction > 0 else bottom
            valid = (rows >= top) & (rows < limit) & (rows < image_height)
            if valid.sum() < len(rows) * 0.65:
                break
            if surface_masks is not None:
                # Each side must be supported by its corresponding model,
                # while the image boundary decides which side may use an overlap.
                eligible = np.flatnonzero(valid)
                mask_support = surface_masks[index][rows[eligible], columns[eligible]]
                if mask_support.mean() < 0.45:
                    break
                valid[eligible] = mask_support
            if spread_samples:
                # Keep support in both the near and far parts of the visible strip.
                visible_offsets = offsets[(rows >= top) & (rows < limit)]
                if not len(visible_offsets):
                    break
                near, far = np.quantile(visible_offsets, (1 / 3, 2 / 3))
                if min(np.count_nonzero(valid & (offsets <= near)),
                       np.count_nonzero(valid & (offsets >= far))) < 12:
                    break
            try:
                planes.append(_fit_samples(
                    depth, columns[valid], rows[valid], offsets[valid],
                    other_boxes, focal, label,
                ))
            except AngleUnavailable:
                break
        if len(planes) != 2:
            continue
        (ramp_normal, ramp_offset, ramp_points), (ground_normal, ground_offset, ground_points) = planes
        angle = _angle_between(ramp_normal, ground_normal)
        # A paint stripe or paver joint on a single plane is not a ramp entrance.
        if not 2.0 <= angle <= 45.0:
            continue
        join_cols = np.linspace(left + width * 0.30, right - width * 0.30, 7)
        rays = pixel_rays(join_cols, slope * join_cols + intercept, image_width, image_height, focal)
        projections = rays @ np.stack((ramp_normal, ground_normal), axis=1)
        if np.any(np.abs(projections) < 1e-8):
            continue
        seam_depths = np.array([ramp_offset, ground_offset]) / projections
        if np.any(seam_depths <= 0):
            continue
        # The two real surfaces should join near the detected boundary.
        separation = np.abs(seam_depths[:, 0] - seam_depths[:, 1]) / seam_depths.mean(axis=1)
        if np.quantile(separation, 0.75) <= 0.03:
            return angle, ramp_points, ground_points
    return None
