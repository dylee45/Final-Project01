# 2주차 개발 기록

## 1. 개발 목표

YOLO 탐지 기능을 개선하고,
실제 이미지 테스트를 통해 모델의 성능과 문제점을 분석한다.

## 2. 경사로 탐지 모델 개선 (팀원 A)

### 구현 내용

- `src/detection.py` 함수 모듈화
- `src/week2_test_detection.py` 다중 이미지 테스트
- 탐지 결과 이미지 및 CSV 저장
- YOLO11n 모델 5 Epoch → 20 Epoch 재학습
- Precision, Recall, mAP 성능 비교

### 모델 평가 결과

| 평가 지표 | 5 Epoch | 20 Epoch |
| --- | ---: | ---: |
| Precision | 0.738 | 0.867 |
| Recall | 0.557 | 0.738 |
| mAP@50 | 0.679 | 0.830 |
| mAP@50-95 | 0.320 | 0.501 |

### 실제 이미지 테스트

총 20장 테스트

- 경사로 사진 10장
- 계단 사진 5장
- 장애물 관련 사진 5장

경사로 탐지 사진 수:
- 5 Epoch: 7/10장
- 20 Epoch: 8/10장

확인된 문제:
- 일부 경사로 미탐지
- 계단을 경사로로 잘못 탐지
- 현재 모델은 ramp 단일 클래스만 학습

## 3. 경사로·계단 모델 시험 학습 (정우혁)

### 구현 내용

- `detection/train.py`: 학습 실행 옵션 구현
- `detection/predict.py`: 이미지 객체 탐지
- 112장 임시 데이터셋 활용
- ramp, stairs 2클래스 학습
- 10 Epoch 및 100 Epoch 결과 비교
- 탐지 이미지 저장 및 신뢰도 출력

### 모델 학습 명령어

```powershell
python detection/train.py --data datasets/final/data.yaml --epochs 100 --batch 8 --name ramp_final
```

### 모델 추론 명령어

```powershell
python detection/predict.py --model runs/detect/ramp_final/weights/best.pt --source datasets/final/test/images --conf 0.25 --name final_predict
```

### 확인된 문제

- 일부 이미지 오탐 발생
- Bounding Box 중복 탐지
- 추가 데이터 확보 및 학습 필요

## 4. GitHub 협업

- `feat/yolo-week02` 브랜치 작업
- YOLO 모델 학습 및 테스트 코드 Commit
- Pull Request #8 생성 및 main 병합
- SourceTree에서 병합 내역 확인

## 5. 향후 작업

1. 경사로·계단·장애물 데이터셋 보강
2. 모델 성능 비교 및 개선
3. 탐지 기능과 통행 조건 판단 로직 연동
4. GUI 및 지도 기능 통합
