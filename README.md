
# 휠체어 사용자를 위한 경사로 탐지 및 통행 조건 분석

## 1. 프로젝트 소개

YOLO Object Detection과 Segmentation을 활용하여
사진 속 경사로와 장애물 후보를 분석하고,
결과를 지도에 표시하는 프로그램입니다.

## 2. GUI 및 지도 시각화
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

이를 통해 학습 횟수 증가로 경사로 탐지 성능은 개선되었지만, 계단과 경사로를 구분하는 데 한계가 있음을 확인하였다. 향후 계단 및 장애물 데이터를 추가하여 탐지 정확도를 개선할 계획이다.

#### 2026.11.18 3주차 정다인

## 4. 사진 기반 경사각 추정

### 구현 내용

- `ramp_angle/estimate.py`: 단일 사진의 깊이 추정 및 경사각 계산
- Depth Anything V2 Metric Outdoor Small 사전 학습 모델 사용
- 경사로와 기준 바닥 영역을 사용자가 직접 선택
- 선택한 진행 방향의 종단 경사각과 두 평면 사이 각도 계산
- 가로 화각 45°, 60°, 75° 조건별 결과 출력 및 JSON 저장

현재 YOLO 탐지 및 GUI와 연동되지 않은 독립 실행 기능입니다.

### 사용 모델 및 각도 계산 방식

- 모델: Depth Anything V2 Metric Outdoor Small
- Hugging Face ID: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`
- 사전 학습된 모델을 추가 학습 없이 사용하여 사진의 깊이를 추정합니다.
- 추정 깊이와 가정한 카메라 화각을 이용해 3차원 좌표를 복원합니다.
- 사용자가 선택한 경사로와 기준 바닥에 각각 평면을 맞춘 뒤,
  선택한 진행 방향의 종단 경사각을 계산합니다.

※ 모델이 경사각을 직접 출력하는 방식이 아니라,
깊이 추정 결과에 기하 계산을 적용하는 방식입니다.

### 개발 환경

- Python
- PyTorch, Transformers
- NumPy, Pillow, Matplotlib

### 실행 방법

**1. 가상환경 생성**

프로젝트 최상위 폴더에서 실행합니다.

```powershell
python -m venv ramp_angle/.venv
```

**2. 라이브러리 설치**

```powershell
.\ramp_angle\.venv\Scripts\python.exe -m pip install -r ramp_angle/requirements.txt
```

**3. 사진 분석**

```powershell
.\ramp_angle\.venv\Scripts\python.exe ramp_angle/estimate.py --image "사진경로.jpg" --out results_test
```

`사진경로.jpg`는 분석할 사진의 실제 경로로 변경합니다.
최초 실행 시 사전 학습 모델을 다운로드하므로 인터넷 연결이 필요합니다.

### 영역 선택 방법

사진 창에서 총 10개의 점을 순서대로 선택합니다.

1. 경사로 바닥 안쪽의 네 점을 사각형 둘레 순서로 선택합니다.
2. 수평으로 확인된 기준 바닥 안쪽의 네 점을 선택합니다.
3. 경사로 영역 안에서 낮은 지점, 높은 지점 순서로 두 점을 선택합니다.

경사로와 기준 바닥 영역은 겹치지 않도록 선택하고,
벽, 난간, 경사로 옆면은 제외합니다.

현재 선택 단계에서는 마우스 오른쪽 버튼으로 마지막 점을 취소할 수 있습니다.
필요한 점 개수를 모두 선택하면 다음 단계로 이동합니다.

### 결과 확인

- `종단 경사각`: 선택한 진행 방향을 따라 계산한 경사각
- `두 평면 사이 각도`: 경사로와 기준 바닥 평면 사이 각도
- 계산 결과는 터미널에 출력되며, 지정한 출력 폴더에 `result.json`으로 저장됩니다.

화각을 직접 지정할 수도 있습니다.

```powershell
.\ramp_angle\.venv\Scripts\python.exe ramp_angle/estimate.py --image "사진경로.jpg" --out results_test --fov 60
```

### 테스트 결과 및 한계

실측 경사각을 확인한 사진 6장으로 테스트한 결과,
5장에서는 실측값과 유사한 추정값을 얻었으며,
정면에서 촬영한 1장에서는 경사각이 크게 추정되었습니다.

이를 통해 촬영 구도에 따라 추정 오차가 달라질 수 있음을 확인했습니다.
향후 정면 사진에서 발생하는 과대 추정 현상을 분석하고 개선할 예정입니다.

※ 소규모 초기 테스트 결과이며, 전체 성능이나 정확도를 의미하지 않습니다.

- 기준 바닥이 수평이라는 가정으로 계산합니다.
- 기본 화각은 실제 사진에서 자동 추출한 값이 아닙니다.
- 현재 결과는 실험적인 추정값이며, 휠체어 통행 가능 여부를 확정하는 기준으로 사용하지 않습니다.
- 기능별 브랜치에서 개발 후 Pull Request로 main에 병합
- 소스 코드와 개발 문서는 GitHub에서 관리
- 데이터셋은 Roboflow 및 로컬 환경에서 관리
- 모델 가중치와 학습 결과는 별도로 백업
