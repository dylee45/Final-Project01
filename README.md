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
## 2-1. Object Detection
Roboflow에서 경사로 라벨링 데이터셋 준비
train.py 작성 및 YOLO11n 모델 학습 (5 Epoch)
detection.py 작성 및 경사로 탐지 테스트
.gitignore 설정으로 데이터셋 및 모델 파일 GitHub 업로드 제외

### 모델 평가 결과

Precision: 0.738

Recall: 0.557

mAP@50: 0.648

## 3-1. YOLO Object Detection 개발 환경



### YOLO Detection 테스트

설치가 완료되면 다음 명령어를 실행합니다.

```powershell
conda activate yolo-project
python detection/test_detection.py
```

테스트 코드는 사전 학습된 YOLO11n 모델을 다운로드하고
샘플 사진에서 객체를 탐지합니다.

정상 실행 시 탐지한 객체의 이름과 신뢰도가 출력됩니다.

### 참고 사항
- 현재 코드는 YOLO Detection 환경 확인용입니다.
- 경사로 전용 탐지 모델은 추후 데이터셋 학습 후 추가할 예정입니다.
- 모델 가중치(`*.pt`), 데이터셋, 학습 결과는 Git 추적에서 제외합니다.
- 공용 PC에서는 작업 종료 전 반드시 Git Push를 완료해야 합니다.


#### 2026.11.18 2주차 정우혁

## 3-2. YOLO Object Detection 모델 학습 및 추론

### 구현 내용

- `train.py`: 사용자 지정 데이터셋을 활용한 YOLO11n 모델 학습
- `predict.py`: 학습된 모델을 활용한 단일 및 다중 이미지 객체 탐지
- 임시 데이터셋 112장(ramp, stairs)을 활용한 시험 학습
- 10 Epoch와 100 Epoch 학습 결과 비교
- 탐지 결과 이미지 자동 저장 및 신뢰도 출력

100 Epoch 학습을 통해 일부 탐지 성능이 개선되었으나,
오탐지 및 중복 탐지 문제는 추가 데이터 학습을 통해 개선할 예정입니다.

### 새로운 데이터셋 학습 방법

**1. 데이터셋 준비**

Object Detection 데이터셋을 YOLOv11 형식으로 다운로드하고
`datasets/final/` 폴더에 압축을 해제합니다.

```text
datasets/final/
├── train/
├── valid/
├── test/
└── data.yaml
```

**2. data.yaml 설정**

사용자 PC의 데이터셋 경로와 클래스 정보를 설정합니다.

```yaml
path: C:/Users/user/Desktop/Final-Project01/datasets/final

train: train/images
val: valid/images
test: test/images

nc: 2
names: ['ramp', 'stairs']
```

※ path는 실제 데이터셋 저장 경로에 맞게 수정하고,
클래스 이름과 순서는 데이터셋의 라벨 정보를 유지합니다.

**3. 모델 학습**

```powershell
conda activate yolo-project
python detection/train.py --data datasets/final/data.yaml --epochs 100 --batch 8 --name ramp_final
```

| 옵션 | 설명 |
| --- | --- |
| --data | 데이터셋 설정 파일 경로 |
| --model | 학습에 사용할 모델 (기본값: yolo11n.pt) |
| --epochs | 학습 횟수 |
| --batch | 배치 크기 |
| --imgsz | 이미지 크기 (기본값: 640) |
| --name | 학습 결과 폴더 이름 |

학습 완료 후 모델은 다음 경로에 저장됩니다.

`runs/detect/ramp_final/weights/best.pt`

### 학습된 모델로 객체 탐지

```powershell
python detection/predict.py --model runs/detect/ramp_final/weights/best.pt --source datasets/final/test/images --conf 0.25 --name final_predict
```

- `--model`: 학습된 모델 파일 경로
- `--source`: 분석할 이미지 또는 폴더 경로
- `--conf`: 최소 탐지 신뢰도
- `--name`: 결과 저장 폴더 이름

탐지 결과는 `runs/predict/final_predict/`에 저장됩니다.
동일한 이름으로 반복 실행할 경우 실제 저장 경로는 터미널에서 확인합니다.

### 기존 모델 추가 학습

기존에 학습된 모델을 사용하여 새로운 데이터셋으로
추가 학습하는 것도 가능합니다.

```powershell
python detection/train.py --data datasets/final/data.yaml --model runs/detect/ramp_final/weights/best.pt --epochs 100 --name ramp_finetuned
```

### 참고 사항

- Python 코드 수정 없이 데이터셋 및 학습 옵션 변경이 가능합니다.
- 데이터셋과 학습된 모델은 Git 추적에서 제외되므로 별도로 공유 및 백업해야 합니다.
- 현재 임시 모델은 기능 검증용이며, 최종 데이터셋을 활용한 재학습을 권장합니다.

**20 Epoch 재학습 및 테스트 결과**

기존 5 Epoch 모델을 20 Epoch로 재학습한 결과, 검증 데이터 기준 mAP50이 약 0.679에서 0.830으로 향상되었다.

실제 사진 20장을 이용한 테스트에서는 경사로 탐지 사진 수가 7장에서 8장으로 증가하였다.

그러나 일부 경사로 사진에서는 탐지에 실패하였으며, `stair04.jpg`에서는 계단을 경사로로 잘못 인식하는 오탐이 발생하였다.

이를 통해 학습 횟수 증가로 경사로 탐지 성능은 개선되었지만, 계단과 경사로를 구분하는 데 한계가 있음을 확인하였다. 향후 계단 및 장애물 데이터를 추가하여 탐지 정확도를 개선할 계획이다.
