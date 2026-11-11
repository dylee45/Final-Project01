# 휠체어 사용자를 위한 경사로 탐지 및 통행 조건 분석

## 1. 프로젝트 소개

YOLO Object Detection과 Segmentation을 활용하여
사진 속 경사로와 장애물 후보를 분석하고,
결과를 지도에 표시하는 프로그램입니다.

## 2. Object Detection



## 2. YOLO Object Detection 개발 환경

### 개발 환경
- Python 3.11
- Conda
- PyTorch 2.6.0 (CUDA 12.6)
- Ultralytics 8.4.210
- 테스트 GPU: NVIDIA GeForce RTX 3060

### 설치 방법 (Windows + NVIDIA GPU)

Anaconda 또는 Miniconda가 설치된 PC에서
프로젝트 루트 디렉터리의 PowerShell을 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

설치 스크립트는 `yolo-project` Conda 환경을 생성하고
필요한 라이브러리를 설치합니다.

이 스크립트는 CUDA 12.6 지원 NVIDIA 드라이버를 사용하는
Windows 환경을 기준으로 작성되었습니다.
다른 환경에서는 PyTorch 설치 방법을 조정해야 할 수 있습니다.

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
