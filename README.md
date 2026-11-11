# 휠체어 사용자를 위한 경사로 탐지 및 통행 조건 분석

## 1. 프로젝트 소개

YOLO Object Detection과 Segmentation을 활용하여
사진 속 경사로와 장애물 후보를 분석하고,
결과를 지도에 표시하는 프로그램입니다.

## 2. 

## 2. GUI 및 지도 시각화

### 2.1 개발 환경
- **Language:** Python
- **GUI Framework:** Streamlit
- **Map Visualization:** Folium, OpenStreetMap
- **Image Processing:** Pillow

### 2.2 구현 기능
- 이미지 업로드 및 미리보기
- 샘플 결과 보여주기
- 지도 표시 및 위치 선택하기

※ 현재 분석 결과는 샘플 데이터이며, 실제 YOLO 및 경사각 추정 모델은 연동되지 않았습니다.

### 2.3 실행 방법

**라이브러리 설치**
```bash
python -m pip install streamlit folium streamlit-folium pillow
```

**GUI 실행**
```bash
python -m streamlit run gui_map/app.py
```

실행 후 브라우저에서 `http://localhost:8501`에 접속합니다.