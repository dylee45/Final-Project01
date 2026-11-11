# 휠체어 사용자를 위한 경사로 탐지 및 통행 조건 분석

## 1. 프로젝트 소개

YOLO Object Detection과 Segmentation을 활용하여
사진 속 경사로와 장애물 후보를 분석하고,
결과를 지도에 표시하는 프로그램입니다.

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
