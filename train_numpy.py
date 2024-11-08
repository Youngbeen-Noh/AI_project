import os
import numpy as np
import cv2
import random

# 상수 정의
LayerLimit = 16
NodeLimit = 256
InputSize = 256 * 256  # 256x256 grayscale 이미지 크기
OutputSize = 4
LearningRate = 0.01  # 학습률 조정

# 레이어 구조 랜덤 설정 함수
def randomize_network():
    layers = random.randint(3, LayerLimit)  # 최소 3개 (input, hidden, output)에서 최대 16개
    nodes_per_layer = [InputSize]  # 첫 레이어는 입력 크기 고정
    for i in range(1, layers - 1):
        nodes_per_layer.append(random.randint(1, NodeLimit))  # 각 hidden layer는 최대 256개의 노드
    nodes_per_layer.append(OutputSize)  # 마지막 레이어는 출력 노드 개수 고정
    return layers, nodes_per_layer

# 모델 설정
class Model:
    def __init__(self, layers, nodes_per_layer):
        self.L = layers
        self.Node = nodes_per_layer

# 랜덤으로 생성된 레이어와 노드 개수로 신경망 구성
layers, nodes_per_layer = randomize_network()
m = Model(layers, nodes_per_layer)

# 가중치 배열 초기화: Xavier 초기화 방식 사용
weight = []
for i in range(m.L - 1):
    weight.append(np.random.randn(m.Node[i], m.Node[i + 1]) * np.sqrt(1 / m.Node[i]))

# Leaky ReLU 함수와 미분 함수
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# 소프트맥스 함수 (출력층 확률화)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Cross-Entropy 손실 함수
def cross_entropy_loss(output, target):
    return -np.sum(target * np.log(output + 1e-9))  # 작은 값 추가로 안전한 log 계산

# 순전파 함수
def forward(layer, input_data, weights):
    return np.dot(input_data, weights[layer])

# 역전파 함수
def backpropagation(input_data, layer_outputs, output, target, weights):
    # 출력층의 오차 및 델타 계산
    output_error = output - target
    delta = output_error * leaky_relu_derivative(output)

    # 역전파를 통한 가중치 업데이트
    for layer in reversed(range(m.L - 1)):
        hidden_layer_output = layer_outputs[layer]
        hidden_error = np.dot(delta, weights[layer].T)
        hidden_delta = hidden_error * leaky_relu_derivative(hidden_layer_output)
        weights[layer] -= LearningRate * np.outer(hidden_layer_output, delta)
        delta = hidden_delta

# 결과 출력 함수
def final_result(output):
    print("\nFinal Result (Probabilities): ")
    print(softmax(output) * 100)  # 확률 값으로 출력

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
            images.append(img)
            label = np.zeros(OutputSize)  # one-hot encoding
            label[label_index] = 1
            labels.append(label)
    return np.array(images), np.array(labels)

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

# 데이터와 라벨을 동물별로 묶어서 배치 형식으로 변환
data = np.array(data)
labels = np.array(labels)
num_animals = len(dataset_folders)

# 학습 루프
num_samples_per_animal = data.shape[1]  # 각 동물별 이미지 수
for epoch in range(1000):
    for i in range(num_samples_per_animal):
        for animal_idx in range(num_animals):
            # 각 동물의 i번째 이미지와 라벨을 가져와서 학습
            input_data = data[animal_idx, i]
            target = labels[animal_idx, i]

            # 순전파
            layer_outputs = [input_data]  # 입력층을 포함하여 각 레이어의 출력을 저장
            for j in range(m.L - 1):
                layer_output = leaky_relu(forward(j, layer_outputs[-1], weight))
                layer_outputs.append(layer_output)
            output_layer = softmax(layer_outputs[-1])

            # 역전파
            backpropagation(input_data, layer_outputs, output_layer, target, weight)

    # 10 epoch마다 손실 및 클래스별 확률 출력
    if epoch % 10 == 0:
        correct_counts = {class_name: 0 for class_name in dataset_folders.keys()}
        
        for animal_idx, animal_data in enumerate(data):
            for sample_idx in range(num_samples_per_animal):
                input_data = animal_data[sample_idx]
                target = labels[animal_idx, sample_idx]
                
                # 순전파
                layer_outputs = [input_data]
                for j in range(m.L - 1):
                    layer_output = leaky_relu(forward(j, layer_outputs[-1], weight))
                    layer_outputs.append(layer_output)
                output_layer = softmax(layer_outputs[-1])
                
                # 예측한 클래스가 타겟과 일치하면 correct_counts 증가
                predicted_class = np.argmax(output_layer)
                true_class = np.argmax(target)
                if predicted_class == true_class:
                    correct_counts[list(dataset_folders.keys())[true_class]] += 1

        # 정확도 및 손실 출력
        accuracy = {class_name: correct_counts[class_name] / num_samples_per_animal * 100 for class_name in correct_counts}
        print(f"Epoch {epoch}: Loss = {cross_entropy_loss(output_layer, target)}")
        print("Class-wise Accuracy:")
        for class_name, acc in accuracy.items():
            print(f"  {class_name}: {acc:.2f}%")

        # 최종 결과 출력 (마지막 샘플의 확률)
        final_result(output_layer)
