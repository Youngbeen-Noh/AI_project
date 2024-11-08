import os
import cv2
import math
import random

# 상수 정의
LayerLimit = 16
InputSize = 256 * 256  # 256x256 grayscale 이미지 크기
OutputSize = 4
LearningRate = 0.01  # 학습률 조정

# 가중치 배열 초기화: Xavier 초기화 방식 사용
def initialize_weights(input_size, output_size):
    return [[(random.random() * 2 - 1) * math.sqrt(1 / input_size) for _ in range(output_size)] for _ in range(input_size)]

weight = [
    initialize_weights(InputSize, 128),  # 입력층에서 은닉층으로
    initialize_weights(128, OutputSize)  # 은닉층에서 출력층으로
]

# 모델 설정
class Model:
    def __init__(self, layers, nodes_per_layer):
        self.L = layers
        self.Node = nodes_per_layer

m = Model(3, [InputSize, 128, OutputSize])

# Leaky ReLU 함수와 미분 함수
def leaky_relu(x, alpha=0.01):
    return [xi if xi > 0 else alpha * xi for xi in x]

def leaky_relu_derivative(x, alpha=0.01):
    return [1 if xi > 0 else alpha for xi in x]

# 소프트맥스 함수 (출력층 확률화)
def softmax(x):
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp_x = sum(exp_x)
    return [xi / sum_exp_x for xi in exp_x]

# Cross-Entropy 손실 함수
def cross_entropy_loss(output, target):
    return -sum(ti * math.log(oi + 1e-9) for ti, oi in zip(target, output))  # 작은 값 추가로 안전한 log 계산

# 순전파 함수 (dot product 구현)
def dot_product(vector_a, matrix_b):
    result = []
    for col in range(len(matrix_b[0])):
        col_sum = 0
        for row in range(len(vector_a)):
            col_sum += vector_a[row] * matrix_b[row][col]
        result.append(col_sum)
    return result

# 역전파 함수
def backpropagation(input_data, hidden_layer_output, output, target, weights):
    # 출력층의 오차 및 델타 계산 (Cross-Entropy 손실 기반)
    output_error = [oi - ti for oi, ti in zip(output, target)]
    delta = [error * deriv for error, deriv in zip(output_error, leaky_relu_derivative(output))]

    # 은닉층의 오차 및 델타 계산
    hidden_error = [sum(delta[i] * weights[1][j][i] for i in range(len(delta))) for j in range(len(hidden_layer_output))]
    hidden_delta = [error * deriv for error, deriv in zip(hidden_error, leaky_relu_derivative(hidden_layer_output))]

    # 가중치 업데이트
    for i in range(len(hidden_layer_output)):
        for j in range(len(delta)):
            weights[1][i][j] -= LearningRate * hidden_layer_output[i] * delta[j]

    for i in range(len(input_data)):
        for j in range(len(hidden_delta)):
            weights[0][i][j] -= LearningRate * input_data[i] * hidden_delta[j]

# 결과 출력 함수
def final_result(output):
    print("\nFinal Result (Probabilities): ")
    print([f"{p * 100:.2f}%" for p in softmax(output)])  # 확률 값으로 출력

# 각 이미지의 경로를 읽고 라벨 생성 함수
def load_images_from_folder(folder, label_index, max_images=100):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(folder)):
        if i >= max_images:  # 100개까지만 로드
            break
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256)).flatten() / 255.0  # 256x256 크기로 조정 후 1D 배열로 변환
            images.append([pixel for pixel in img])
            label = [0] * OutputSize  # one-hot encoding
            label[label_index] = 1
            labels.append(label)
    return images, labels

# 학습할 이미지 데이터셋 경로
dataset_folders = {
    "dog": "dataset/dog/train",
    "cat": "dataset/cat/train",
    "hyena": "dataset/hyena/train",
    "tiger": "dataset/tiger/train"
}

# 각 폴더의 이미지와 라벨을 로드
data = []
labels = []
for index, (label, folder) in enumerate(dataset_folders.items()):
    images, folder_labels = load_images_from_folder(folder, index)
    data.append(images)
    labels.append(folder_labels)

# 데이터와 라벨을 동물별로 분리한 리스트로 저장
data_by_class = data
labels_by_class = labels
num_classes = len(data_by_class)

# 학습 루프
for epoch in range(1000):
    for i in range(len(data_by_class[0])):  # 각 클래스의 100개 이미지를 하나씩 순서대로 사용
        for class_idx in range(num_classes):
            input_data = data_by_class[class_idx][i]
            target = labels_by_class[class_idx][i]

            # 순전파
            hidden_layer_output = leaky_relu(dot_product(input_data, weight[0]))
            output_layer = softmax(dot_product(hidden_layer_output, weight[1]))

            # 역전파
            backpropagation(input_data, hidden_layer_output, output_layer, target, weight)

    # 주기적으로 출력
    print(f"Epoch {epoch}: Loss = {cross_entropy_loss(output_layer, target):.4f}")
    final_result(output_layer)
