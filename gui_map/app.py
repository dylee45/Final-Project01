import sys
from pathlib import Path

import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image, ImageOps, UnidentifiedImageError

# Streamlit executes this file from gui_map, while project modules live one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis import analyze_image, resolve_weights
from src.detection import RampSegmentation, load_model


@st.cache_resource
def cached_model(path, modified_time):
    return load_model(path)


def clear_analysis():
    st.session_state.analysis_result = None

# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="휠체어 통행 환경 분석",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# 2. CSS 디자인
# --------------------------------------------------

st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background-color: #F6F8F5;
}

/* 기본 글씨 */
html, body {
    font-family: 'Pretendard', sans-serif;
    color: #263B33;
}

/* Streamlit 기본 헤더 아래에 충분한 공간 확보 */
.block-container {
    padding-top: 5rem !important;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* 제목 */
.main-title {
    font-size: 38px;
    font-weight: 750;
    color: #263B33;
    margin-bottom: 12px;
}

/* 부제목 */
.subtitle {
    font-size: 16px;
    color: #71847A;
    margin-bottom: 30px;
}

/* 카드 제목 */
.card-title {
    font-size: 21px;
    font-weight: 700;
    color: #263B33;
    margin-bottom: 20px;
}

/* 버튼 */
.stButton > button {
    background-color: #3D6B54;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #2F5945;
    color: white;
}

/* 파일 업로드 */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}

/* 입력창 */
.stTextInput input {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Session State 초기화
# --------------------------------------------------

if "latitude" not in st.session_state:
    st.session_state.latitude = 36.659727

if "longitude" not in st.session_state:
    st.session_state.longitude = 126.896189

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "map_results" not in st.session_state:
    st.session_state.map_results = {}


# --------------------------------------------------
# 4. 상단 헤더
# --------------------------------------------------

header_left, header_right = st.columns([8, 1])

with header_left:

    st.caption("사진으로 살펴보는 이동 환경")

    st.markdown(
        '<div class="main-title">어떤 길인지, 먼저 확인하세요.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        '사진과 위치를 선택하고 접근성 정보를 확인해 보세요.'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:

    if st.button("처음부터"):
        clear_analysis()
        st.rerun()


# --------------------------------------------------
# 5. 메인 화면 (왼쪽 / 오른쪽)
# --------------------------------------------------

left, right = st.columns([1, 1.2], gap="large")


# ==================================================
# 왼쪽 : 이미지 분석
# ==================================================

with left:
    with st.container(border=True):
        st.markdown("### 현장 사진")
        uploaded_file = st.file_uploader(
            "분석할 사진을 업로드하세요.",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            on_change=clear_analysis,
        )
        image = None
        if uploaded_file is None:
            st.info("분석할 사진을 업로드해 주세요.")
        else:
            try:
                with Image.open(uploaded_file) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
                st.error("사진을 열 수 없습니다. 다른 사진을 업로드해 주세요.")
            if image is not None:
                st.image(image, caption=uploaded_file.name, width="stretch")

        with st.expander("분석 설정"):
            weights = st.text_input(
                "YOLO 모델 경로",
                placeholder="best.pt",
                on_change=clear_analysis,
            )
            ground_weights = st.text_input(
                "기준 바닥 모델 경로 (선택)",
                placeholder="runs/segment/ramp_seg/weights/best.pt",
                on_change=clear_analysis,
            ).strip()
            ramp_segment_weights = st.text_input(
                "경사로 분할 모델 경로 (선택)",
                placeholder="runs/segment/ramp_mask/weights/best.pt",
                on_change=clear_analysis,
            ).strip()
            ramp_confidence = None
            spread_samples = False
            if ramp_segment_weights:
                st.caption("경사로를 탐지한 뒤 원본 사진에서 경사면 마스크를 찾습니다. 탐지 박스와 맞는 마스크가 없으면 박스 주변을 다시 검사합니다.")
                ramp_confidence = st.number_input(
                    "경사면 분할 신뢰도 기준 (%)", min_value=1, max_value=100,
                    value=20, on_change=clear_analysis,
                ) / 100
                spread_samples = st.checkbox("표본 범위 확장", on_change=clear_analysis)
                if not ground_weights:
                    st.warning("경사로 분할과 각도 추정을 위해 기준 바닥 모델 경로도 입력해 주세요.")
            elif ground_weights:
                st.caption("첫 번째 모델에서 경사로, 기준 바닥 모델에서 바닥과 계단·장애물을 가져옵니다.")
            fov = st.number_input(
                "카메라 가로 화각 (°)", min_value=1.0, max_value=179.0,
                value=60.0, on_change=clear_analysis,
            )
            st.caption("기본 화각 60°는 추정을 위한 가정입니다.")
            use_limit = st.checkbox("프로젝트 경사 기준 사용", on_change=clear_analysis)
            limit = None
            if use_limit:
                limit = st.number_input(
                    "프로젝트 경사 기준 (°)", min_value=0.0, max_value=90.0,
                    value=None, placeholder="정한 기준을 입력하세요",
                    on_change=clear_analysis,
                )

        if st.button(
            "사진 분석 실행",
            disabled=image is None or (use_limit and limit is None)
            or bool(ramp_segment_weights and not ground_weights),
            width="stretch",
        ):
            clear_analysis()
            try:
                with st.spinner("사진에서 경사로와 기준 바닥을 찾는 중입니다."):
                    path = resolve_weights(weights)
                    model = cached_model(str(path), path.stat().st_mtime_ns)
                    ground_model = None
                    if ground_weights:
                        ground_path = resolve_weights(ground_weights)
                        ground_model = cached_model(str(ground_path), ground_path.stat().st_mtime_ns)
                    ramp_segment_model = None
                    if ramp_segment_weights:
                        ramp_segment_path = resolve_weights(ramp_segment_weights)
                        ramp_segment_model = RampSegmentation(
                            cached_model(str(ramp_segment_path), ramp_segment_path.stat().st_mtime_ns),
                            ramp_confidence,
                        )
                    st.session_state.analysis_result = analyze_image(
                        image, model, fov, limit,
                        ground_model=ground_model, ramp_segment_model=ramp_segment_model,
                        spread_samples=spread_samples,
                    )
            except Exception as error:
                st.error(f"사진 분석을 마치지 못했습니다. {error}")

        result = st.session_state.analysis_result
        if result is not None:
            st.divider()
            st.markdown("### 분석 결과")
            st.write(f"**{result['status']}** · {result['reason']}")
            st.image(result["preview"], caption="YOLO 탐지 및 자동 표본", width="stretch")
            if any(ramp["angle"] is not None for ramp in result["ramps"]):
                st.caption("하늘색 점은 경사로, 주황색 점은 기준 바닥에 사용한 표본입니다.")
            for index, ramp in enumerate(result["ramps"], start=1):
                if ramp["angle"] is None:
                    st.warning(f"경사로 {index} 추정 불가 · {ramp['reason']}")
                else:
                    st.write(f"경사로 {index} 추정 경사각 · {ramp['angle']:.2f}°")
            if result["detections"]:
                st.caption("탐지 객체 · " + ", ".join(
                    f"{obj['class_name']} ({obj['confidence']:.0%})"
                    for obj in result["detections"]
                ))

with right:
    with st.container(border=True):
        st.markdown("### 촬영 위치")
        st.caption("지도에서 위치를 고르고 분석 결과를 표시하세요.")
        location = (st.session_state.latitude, st.session_state.longitude)
        m = folium.Map(location=location, zoom_start=16)
        if location not in st.session_state.map_results:
            folium.Marker(
                location=location, tooltip="선택한 촬영 위치",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
        colors = {
            "경사 기준 이내": "green",
            "경사 기준 초과": "orange",
            "장애물 확인 필요": "red",
            "확인 필요": "gray",
        }
        for coords, saved in st.session_state.map_results.items():
            folium.Marker(
                location=coords,
                tooltip=saved["status"],
                popup=" · ".join([saved["status"], *saved["angles"]]),
                icon=folium.Icon(color=colors[saved["status"]], icon="info-sign"),
            ).add_to(m)

        map_data = st_folium(
            m, height=450, use_container_width=True, key="map",
            returned_objects=["last_clicked"],
        )
        if map_data and map_data.get("last_clicked"):
            click = map_data["last_clicked"]
            coords = (click["lat"], click["lng"])
            if coords != st.session_state.get("last_map_click"):
                st.session_state.last_map_click = coords
                st.session_state.latitude, st.session_state.longitude = coords
                st.session_state.pop("latitude_input", None)
                st.session_state.pop("longitude_input", None)
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input(
                "위도", min_value=-90.0, max_value=90.0,
                value=location[0], format="%.6f", key="latitude_input",
            )
        with col2:
            longitude = st.number_input(
                "경도", min_value=-180.0, max_value=180.0,
                value=location[1], format="%.6f", key="longitude_input",
            )
        if st.button("위치 적용", width="stretch"):
            st.session_state.latitude = latitude
            st.session_state.longitude = longitude
            st.rerun()

        st.caption(f"선택한 위치 · {location[0]:.6f}, {location[1]:.6f}")
        if st.button("지도에 분석 결과 표시", disabled=result is None, width="stretch"):
            st.session_state.map_results[location] = {
                "status": result["status"],
                "angles": [
                    f"경사로 {index} {ramp['angle']:.2f}°" if ramp["angle"] is not None
                    else f"경사로 {index} 추정 불가"
                    for index, ramp in enumerate(result["ramps"], start=1)
                ],
            }
            st.rerun()
