import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

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
        st.session_state.analysis_result = None
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
            label_visibility="collapsed"
        )

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

        else:

            st.info("분석할 사진을 업로드해 주세요.")

        st.write("")

        st.markdown("#### 샘플 결과 유형")

        sample_type = st.selectbox(
            "분석 유형 선택",
            [
                "통행 가능한 경사로",
                "경사가 높은 경사로",
                "계단이 있는 환경",
                "장애물이 있는 환경"
            ],
            label_visibility="collapsed"
        )

        st.caption(
            "현재는 샘플 데이터를 사용합니다. "
            "실제 YOLO 분석은 추후 연동됩니다."
        )

        st.write("")

        if st.button(
            "샘플 분석 실행 ↗",
            use_container_width=True
        ):

            sample_results = {

                "통행 가능한 경사로": {
                    "status": "통행 가능",
                    "angle": 3.5,
                    "reason": "샘플 데이터상 통행 가능한 경사로입니다."
                },

                "경사가 높은 경사로": {
                    "status": "통행 주의",
                    "angle": 9.2,
                    "reason": "샘플 데이터상 경사가 높은 구간입니다."
                },

                "계단이 있는 환경": {
                    "status": "통행 불가능",
                    "angle": None,
                    "reason": "계단이 탐지된 상황을 가정한 결과입니다."
                },

                "장애물이 있는 환경": {
                    "status": "통행 불가능",
                    "angle": None,
                    "reason": "장애물이 탐지된 상황을 가정한 결과입니다."
                }
            }

            st.session_state.analysis_result = sample_results[sample_type]

        # ------------------------------------------
        # 분석 결과 출력
        # ------------------------------------------

        if st.session_state.analysis_result is not None:

            result = st.session_state.analysis_result

            st.divider()

            st.markdown("### 분석 결과")

            status = result["status"]

            if status == "통행 가능":

                st.success("🟢 통행 가능")

            elif status == "통행 주의":

                st.warning("🟡 통행 주의")

            else:

                st.error("🔴 통행 불가능")

            if result["angle"] is not None:

                st.metric(
                    label="경사각",
                    value=f'{result["angle"]}°'
                )

            st.write(result["reason"])

            st.caption("※ 실제 AI 분석 결과가 아닌 샘플 데이터입니다.")


# ==================================================
# 오른쪽 : 지도
# ==================================================

with right:

    with st.container(border=True):

        st.markdown("### 촬영 위치")

        st.caption("지도를 클릭하여 위치를 선택하세요.")

        # 지도 생성
        m = folium.Map(
            location=[
                st.session_state.latitude,
                st.session_state.longitude
            ],
            zoom_start=16
        )

        # 현재 선택된 위치 마커
        folium.Marker(
            location=[
                st.session_state.latitude,
                st.session_state.longitude
            ],
            tooltip="선택한 촬영 위치",
            icon=folium.Icon(
                color="green",
                icon="info-sign"
            )
        ).add_to(m)

        # Streamlit에 지도 출력
        map_data = st_folium(
            m,
            height=450,
            use_container_width=True,
            key="map"
        )

        # 지도 클릭 시 좌표 저장
        if map_data and map_data.get("last_clicked"):

            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]

            st.session_state.latitude = clicked_lat
            st.session_state.longitude = clicked_lon

        st.write("")

        # 위도 / 경도 입력
        col1, col2 = st.columns(2)

        with col1:

            latitude = st.number_input(
                "위도",
                value=st.session_state.latitude,
                format="%.6f"
            )

        with col2:

            longitude = st.number_input(
                "경도",
                value=st.session_state.longitude,
                format="%.6f"
            )

        if st.button(
            "위치 적용",
            use_container_width=True
        ):

            st.session_state.latitude = latitude
            st.session_state.longitude = longitude

            st.rerun()

        st.caption(
            f"선택한 위치 · "
            f"{st.session_state.latitude:.6f}, "
            f"{st.session_state.longitude:.6f}"
        )