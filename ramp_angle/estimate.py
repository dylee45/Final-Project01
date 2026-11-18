import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def select_regions(image):
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(image)
    ax.set_axis_off()

    def click_points(count, title):
        ax.set_title(title)
        fig.canvas.draw_idle()
        points = fig.ginput(count, timeout=0, show_clicks=True)
        if len(points) != count:
            raise ValueError(f"점 {count}개를 선택해야 합니다.")
        return points

    def select_polygon(title, color):
        points = click_points(4, title)
        polygon = np.array(points + [points[0]])
        ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=2)

        mask_image = Image.new("L", image.size)
        ImageDraw.Draw(mask_image).polygon(points, fill=255)
        return np.asarray(mask_image) > 0

    try:
        ramp_mask = select_polygon(
            "1. Click 4 corners inside the ramp surface", "cyan"
        )
        ground_mask = select_polygon(
            "2. Click 4 corners inside the level ground", "orange"
        )
        endpoints = click_points(
            2, "3. Click lower point, then upper point along the ramp"
        )

        for x, y in endpoints:
            col, row = round(x), round(y)
            if not (
                0 <= col < image.width
                and 0 <= row < image.height
                and ramp_mask[row, col]
            ):
                raise ValueError("시작점과 끝점을 모두 경사로 영역 안에 찍어 주세요.")

        return ramp_mask, ground_mask, endpoints
    finally:
        plt.close(fig)


def estimate_depth(image):
    print("깊이 모델 불러오는 중...", flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).eval()
    inputs = processor(images=image, return_tensors="pt")

    print("깊이 추정 중...", flush=True)
    with torch.inference_mode():
        depth = model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]

    return depth.cpu().numpy()


def fit_plane(points):
    if len(points) < 30:
        raise ValueError("평면을 맞추기에 유효한 깊이 점이 부족합니다.")

    active = points
    for _ in range(4):
        center = active.mean(axis=0)
        _, _, vectors = np.linalg.svd(active - center, full_matrices=False)
        normal = vectors[-1]
        residuals = np.abs((points - center) @ normal)
        active = points[residuals <= np.quantile(residuals, 0.85)]

    center = active.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(
        active - center, full_matrices=False
    )
    if singular_values[1] < 1e-8:
        raise ValueError("선택한 점들이 한 줄에 몰려 평면을 구할 수 없습니다.")

    return center, vectors[-1]


def pixel_rays(x, y, width, height, focal):
    return np.stack(
        (
            (x - width / 2) / focal,
            (y - height / 2) / focal,
            np.ones_like(x),
        ),
        axis=-1,
    )


def calculate_angles(depth, ramp_mask, ground_mask, endpoints, fov):
    height, width = depth.shape
    focal = width / (2 * np.tan(np.radians(fov / 2)))
    valid = np.isfinite(depth) & (depth > 0)

    def points_in(mask):
        rows, cols = np.nonzero(mask & valid)
        rays = pixel_rays(cols, rows, width, height, focal)
        return rays * depth[rows, cols, None]

    ramp_center, ramp_normal = fit_plane(points_in(ramp_mask))
    ground_center, ground_normal = fit_plane(points_in(ground_mask))

    # 바닥 평면의 법선이 카메라 쪽을 향하도록 맞춘다.
    # 촬영자가 바닥 위에 있다는 가정에서 이 방향이 위쪽이다.
    up = ground_normal
    if up @ ground_center > 0:
        up = -up

    # 클릭한 두 픽셀의 시선이 경사로 평면과 만나는 3D 점을 구한다.
    pixels = np.asarray(endpoints, dtype=float)
    rays = pixel_rays(pixels[:, 0], pixels[:, 1], width, height, focal)
    denominator = rays @ ramp_normal

    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("클릭한 지점의 시선이 경사로 평면과 거의 평행합니다.")

    ray_depths = (ramp_center @ ramp_normal) / denominator
    if np.any(~np.isfinite(ray_depths) | (ray_depths <= 0)):
        raise ValueError("클릭한 지점의 3D 위치를 복원할 수 없습니다.")

    lower_point, upper_point = rays * ray_depths[:, None]
    displacement = upper_point - lower_point
    rise = float(displacement @ up)
    run = float(np.linalg.norm(displacement - rise * up))

    if run < 1e-8:
        raise ValueError("두 지점 사이의 수평거리가 너무 짧습니다.")

    longitudinal_angle = np.degrees(np.arctan2(rise, run))
    plane_angle = np.degrees(
        np.arccos(np.clip(abs(ramp_normal @ ground_normal), 0, 1))
    )

    return {
        "horizontal_fov_deg": fov,
        "longitudinal_slope_deg": float(longitudinal_angle),
        "plane_angle_deg": float(plane_angle),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="results_longitudinal")
    parser.add_argument("--fov", type=float, nargs="+", default=[45, 60, 75])
    args = parser.parse_args()

    if any(not 0 < fov < 180 for fov in args.fov):
        parser.error("--fov는 0보다 크고 180보다 작아야 합니다.")

    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((1024, 1024))

    try:
        ramp_mask, ground_mask, endpoints = select_regions(image)
        if np.any(ramp_mask & ground_mask):
            raise ValueError("경사로와 기준 바닥 영역이 겹칩니다.")

        depth = estimate_depth(image)
        results = []
        for fov in args.fov:
            row = calculate_angles(depth, ramp_mask, ground_mask, endpoints, fov)
            results.append(row)
            print(
                f"가로 화각 {fov:g}° | "
                f"종단 경사각 {row['longitudinal_slope_deg']:+.2f}° | "
                f"두 평면 사이 각도 {row['plane_angle_deg']:.2f}°"
            )
    except ValueError as error:
        parser.exit(1, f"측정 실패: {error}\n")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "image": args.image,
        "reference": "selected ground plane assumed level",
        "endpoints_px_lower_to_upper": endpoints,
        "results": results,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"저장 완료: {result_path}")


if __name__ == "__main__":
    main()