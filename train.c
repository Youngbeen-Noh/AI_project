#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define LayerLimit 16
#define NodeLimit 256
#define InputSize 256*256  // 입력 크기 정의 (256x256)
#define OutputSize 4       // 출력 노드 수 (개, 고양이, 호랑이, 하이에나)
#define LearningRate 0.01  // 학습률

// 가중치 배열
float weight[LayerLimit][NodeLimit][NodeLimit];

// 레이어 개수와 레이어별 노드 개수를 저장할 구조체
typedef struct {
    int L;
    int Node[LayerLimit];
} Model;

// 구조체 생성
Model m;

// 시그모이드 계산 함수
float sigmoid(float _x) {
    return 1.0f / (1.0f + exp(-_x));
}

// 시그모이드 미분 계산 함수 (역전파에 필요)
float sigmoid_derivative(float _x) {
    return _x * (1 - _x);
}

// 순전파 함수
float forward(int layer, int node, float input[NodeLimit], float w[LayerLimit][NodeLimit][NodeLimit]) {
    float result = 0;
    for (int k = 0; k < m.Node[layer]; k++) {
        result += input[k] * w[layer][k][node];
    }
    return result;
}

// 결과 출력 함수
void final_result(float input[OutputSize]) {
    printf("\nFinal Result : \n");
    for (int i = 0; i < OutputSize; i++) {
        printf("%f ", input[i]);
    }
    printf("\n");
}

// 역전파 함수
void backpropagation(float input[NodeLimit], float output[NodeLimit], float target[OutputSize], float weight[LayerLimit][NodeLimit][NodeLimit]) {
    float error[NodeLimit];
    float delta[NodeLimit];

    // 출력 레이어의 오차 및 델타 계산
    for (int i = 0; i < OutputSize; i++) {
        error[i] = target[i] - output[i];
        delta[i] = error[i] * sigmoid_derivative(output[i]);
    }

    // 역전파를 통한 가중치 업데이트
    for (int layer = m.L - 2; layer >= 0; layer--) {
        float prev_delta[NodeLimit];
        for (int j = 0; j < m.Node[layer]; j++) {
            prev_delta[j] = 0;
            for (int i = 0; i < m.Node[layer + 1]; i++) {
                prev_delta[j] += delta[i] * weight[layer][j][i];
                weight[layer][j][i] += LearningRate * delta[i] * input[j];
            }
            prev_delta[j] *= sigmoid_derivative(input[j]);
        }
        for (int i = 0; i < m.Node[layer]; i++) {
            delta[i] = prev_delta[i];
        }
    }
}

int main() {
    srand(time(NULL));

    // 네트워크 설정
    m.L = 3;          // 레이어 개수
    m.Node[0] = InputSize;  // 입력 노드
    m.Node[1] = 128;        // 은닉층 노드
    m.Node[2] = OutputSize; // 출력 노드

    float input[NodeLimit] = {0};   // 입력 노드
    float output[NodeLimit] = {0};  // 출력 노드
    float target[OutputSize] = {1, 0, 0, 0};  // 예시 타겟 값 (개)

    // 가중치 초기화 (임의의 값으로)
    for (int i = 0; i < m.L - 1; i++) {
        for (int j = 0; j < m.Node[i]; j++) {
            for (int k = 0; k < m.Node[i + 1]; k++) {
                weight[i][j][k] = ((float)rand() / RAND_MAX) * 0.1;
            }
        }
    }

    // 학습 루프
    for (int epoch = 0; epoch < 1000; epoch++) {
        // 순전파
        for (int i = 0; i < m.L - 1; i++) {
            for (int j = 0; j < m.Node[i + 1]; j++) {
                output[j] = forward(i, j, input, weight);
                output[j] = sigmoid(output[j]);
            }
            for (int j = 0; j < m.Node[i + 1]; j++) {
                input[j] = output[j];
            }
        }

        // 역전파
        backpropagation(input, output, target, weight);

        // 출력
        if (epoch % 100 == 0) {
            printf("Epoch %d: ", epoch);
            final_result(output);
        }
    }

    return 0;
}
