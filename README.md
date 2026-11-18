
# 휠체어 사용자를 위한 경사로 탐지 및 통행 조건 분석

## 1. 프로젝트 소개

본 프로젝트는 YOLO11 Object Detection을 활용하여
사진 속 경사로, 계단 및 장애물 후보를 분석하고,
휠체어 통행 가능 여부를 판단하여 지도에 시각화하는
프로그램 개발을 목표로 합니다.

현재는 경사로 탐지 모델과 GUI 기본 기능을 구현했으며,
통행 조건 판단 및 기능 통합을 진행할 예정입니다.

## 2. 개발 환경

| 구분 | 사용 기술 |
| --- | --- |
| Language | Python |
| Object Detection | YOLO11n |
| Deep Learning | PyTorch |
| Dataset | Roboflow |
| GUI | Streamlit |
| Map | Folium, OpenStreetMap |
| Image Processing | Pillow |
| Version Control | Git, GitHub, SourceTree |

## 3. 주요 기능 및 진행 현황

- 경사로 객체 탐지 모델 학습 및 테스트
- 경사로·계단 2클래스 모델 시험 학습
- 다중 이미지 탐지 및 결과 저장
- 이미지 업로드 및 미리보기
- 지도 표시 및 위치 선택
- 통행 조건 판단 및 실제 YOLO-GUI 연동은 개발 예정

## 4. 실행 방법

### GUI 실행

```bash
python -m pip install -r requirements.txt
python -m streamlit run gui_map/app.py
```

### YOLO 기본 환경 테스트

```bash
python detection/test_detection.py
```

### 경사로 탐지 모델 학습

```bash
python src/train.py
```

### 학습된 모델로 객체 탐지

```bash
python detection/predict.py --model runs/detect/ramp_final/weights/best.pt --source datasets/final/test/images
```

※ 학습 및 탐지 명령어는 필요한 데이터셋과 모델 파일이
로컬에 준비된 경우 실행할 수 있습니다.

## 5. 개발 진행 기록

- [1주차 개발 기록](docs/week01.md)
- [2주차 개발 기록](docs/week02.md)

## 6. 향후 개발 계획

1. 경사로·계단·장애물 데이터셋 보강
2. 모델 성능 개선 및 오탐·미탐 분석
3. 통행 조건 판단 로직 구현
4. YOLO 탐지 결과와 GUI 연동
5. 지도 시각화 및 통합 테스트

## 7. GitHub 및 데이터 관리

- 기능별 브랜치에서 개발 후 Pull Request로 main에 병합
- 소스 코드와 개발 문서는 GitHub에서 관리
- 데이터셋은 Roboflow 및 로컬 환경에서 관리
- 모델 가중치와 학습 결과는 별도로 백업
