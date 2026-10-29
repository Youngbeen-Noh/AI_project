# CUDA로 직접 구현한 MLP 이미지 분류기

딥러닝 프레임워크 없이 CUDA C++로 다층 퍼셉트론(MLP)의 순전파·역전파·가중치 초기화를 직접 구현해, 개·고양이·하이에나·호랑이 4종 이미지를 분류한 개인 프로젝트입니다. (인천대학교 인공지능 수업, 2024.10 – 11)

## 구현 단계

같은 신경망을 네 단계로 다시 구현하며 연산 구조를 익혔습니다.

| 단계 | 파일 | 내용 |
|---|---|---|
| 1 | `train.c` | C로 작성한 CPU 버전 (sigmoid) |
| 2 | `train.py` | 순수 Python 구현 |
| 3 | `train_numpy.py` | NumPy 벡터화 |
| 4 | `trainCase1.cu` ~ `trainCase4.cu` | CUDA 커널로 GPU 구현 |

## 네트워크와 실험

- 입력: 256×256 그레이스케일 이미지를 임계값 128로 이진화한 65,536차원 벡터
- 학습: Leaky ReLU, Softmax, Cross-Entropy, 기울기 클리핑(±1), 배치 1 SGD, 100 epoch

| 실험 | 구조 | 학습률 |
|---|---|---|
| Case 1 | 8층 (65536-256-128-64-64-128-256-4) | 1e-5 |
| Case 2 | 8층 (동일) | 1e-6 |
| Case 3 | 4층 (65536-32-256-4) | 1e-5 |
| Case 4 | 4층 (동일) | 1e-6 |

## CUDA 커널

- `initialize_weights`: cuRAND 정규분포 초기화 (표준편차 √(1/n))
- `forward_propagation`: 출력 뉴런마다 스레드 1개가 행렬-벡터 곱 수행
- `backward_propagation_output`, `backward_propagation_hidden`: 출력층·은닉층 역전파
- 디바이스 함수: `leaky_relu`와 그 미분, `softmax`

## 데이터

- 클래스당 최대 4,000장, 학습:테스트 = 8:2, 클래스를 번갈아 배치해 균형 유지
- `convert_jpg.py`: 256×256 그레이스케일 변환
- `addNoise.py`: 데이터가 적은 하이에나 클래스에 가우시안·소금-후추·스페클 노이즈로 증강
- 데이터셋은 용량 문제로 저장소에 포함하지 않았습니다.

## 결과 기록

학습 중 epoch마다 train/test loss와 클래스별 정확도를 로그 파일에 기록하고, `make_result.py`로 Case별 loss 곡선과 정확도 그래프를 그립니다. 데이터셋과 결과 파일은 용량 문제로 저장소에 포함하지 않았습니다.

## 빌드

- nvcc, C++17, OpenCV 4.10, MSVC 2019 (`.vscode/tasks.json` 참고)

## 알려진 한계와 개선 방향

- 출력층 역전파에서 은닉층으로 넘기는 오차(`hidden_errors`)를 여러 스레드가 atomic 없이 누적해 경쟁 상태가 생길 수 있습니다. → `atomicAdd` 또는 리덕션 커널로 개선 예정
- 순전파 커널 안의 softmax가 블록 간 동기화 없이 다른 스레드의 값을 읽습니다. → 별도 커널로 분리 예정
- 은닉층으로 오차를 전달할 때 활성화 함수 미분이 빠져 있습니다.
- 행렬-벡터 곱이 기본(naive) 구현이라 shared memory 타일링을 적용할 여지가 있습니다.
