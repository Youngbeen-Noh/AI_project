#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <filesystem>
#define CUDA_CHECK(call)                                                       \
{                                                                              \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "  \
                  << "code=" << err << " (" << cudaGetErrorString(err) << ")"  \
                  << std::endl;                                                \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define LayerLimit 16
#define NodeLimit 256
#define InputSize 256 * 256 // 65536 노드 (256x256 이미지)
#define OutputSize 4
#define LearningRate 0.00001

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// 활성화 함수 (Leaky ReLU)
__device__ float leaky_relu(float x) {
    return x > 0 ? x : 0.01f * x;
}

__device__ float leaky_relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.01f;
}

// 소프트맥스 함수
__device__ void softmax(float* input, float* output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum_exp;
        output[i] = fmaxf(fminf(output[i], 1.0f), 1e-9f);
    }
}

// 교차 엔트로피 손실 함수
__host__ float cross_entropy_loss(float* output, float* target, float* node_errors, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        // 크로스 엔트로피 손실 계산
        node_errors[i] = -target[i] * logf(fmaxf(output[i], 1e-9f)); // 노드별 오차 저장
        loss += node_errors[i];
    }
    return loss;
}

// 가중치 초기화 함수 (He, Xavier) - 고정된 시드 사용
__global__ void initialize_weights(float* weights, int input_size, int output_size, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size * output_size) {
        curandState state;
        curand_init(seed, idx, 0, &state); // 고정된 시드 사용
        float stddev = sqrtf(1.0f / input_size); // He 초기화
        weights[idx] = curand_normal(&state) * stddev;
    }
}


// 순전파 함수
__global__ void forward_propagation(float* before_output, float* weights, float* r_input, float* r_output, int input_size, int output_size, bool is_last) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += before_output[i] * weights[i * output_size + idx];
        }
        r_input[idx] = sum; // 활성화 전 값 저장
        if(is_last){
            softmax(r_input, r_output, output_size);
        }
        else{
            r_output[idx] = leaky_relu(sum);
        }
    }
}

// 역전파 - 출력층의 오차 계산 및 가중치 업데이트
__global__ void backward_propagation_output(
    float* hidden_errors, float* output_errors, 
    float* output, float* target, 
    float* weights, float* hidden_output, 
    int hidden_size, float learning_rate = LearningRate) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < OutputSize) {
        // 출력층 오차 계산 (Softmax + Cross Entropy)
        output_errors[idx] = output[idx] - target[idx];
        
        // 은닉층 오차 초기화
        for (int j = 0; j < hidden_size; j++) {
            hidden_errors[j] = 0.0f; // 은닉층 오차 초기화
        }

        // 은닉층으로 오차 전파 및 가중치 업데이트
        for (int j = 0; j < hidden_size; j++) {
            int weight_idx = j * OutputSize + idx;
            
            // 은닉층의 오차 누적
            hidden_errors[j] += output_errors[idx] * weights[weight_idx];

            // 출력층 가중치 업데이트 (기울기 클리핑 추가)
            float gradient = output_errors[idx] * hidden_output[j];
            gradient = fminf(fmaxf(gradient, -1.0f), 1.0f); // 기울기 클리핑
            weights[weight_idx] -= learning_rate * gradient;
        }
    }
}

// 역전파 - 은닉층의 오차 계산 및 가중치 업데이트
__global__ void backward_propagation_hidden(
    float* layer_errors, float* next_layer_errors, 
    float* from_weights, float* layer_weights, 
    float* from_outputs, float* layer_input, 
    int from_layer_size, int layer_size, int next_layer_size, 
    float learning_rate = LearningRate) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < layer_size) {
        float error = 0.0f;

        // 현재 레이어 오차 초기화
        layer_errors[idx] = 0.0f;

        // 다음 레이어로부터 오차를 전파받아 은닉층의 오차 계산
        for (int i = 0; i < next_layer_size; i++) {
            error += next_layer_errors[i] * layer_weights[idx * next_layer_size + i];
        }
        // 현재 레이어 오차 저장
        layer_errors[idx] = error;

        // Leaky ReLU의 미분 적용
        float derivative = leaky_relu_derivative(layer_input[idx]);
        error *= derivative;

        // 은닉층 가중치 업데이트
        for (int j = 0; j < from_layer_size; j++) {
            int weight_idx = j * layer_size + idx;
            float gradient = error * from_outputs[j];

            // 기울기 클리핑 적용
            gradient = fminf(fmaxf(gradient, -1.0f), 1.0f);
            from_weights[weight_idx] -= learning_rate * gradient;
        }

    }
}

// 이미지를 읽어와 데이터 셋을 만드는 함수
vector<pair<vector<float>, vector<float>>> load_images(const string& folder_path, int label_index, int max_images = 4000) {
    vector<pair<vector<float>, vector<float>>> dataset;
    int image_count = 0; // 읽어온 이미지 수를 추적

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (image_count >= max_images) break;

        // 이미지를 그레이스케일로 읽기
        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        // 그레이스케일 이미지를 이진화(흑백 변환)
        Mat binary_img;
        double threshold_value = 128; // 임계값
        threshold(img, binary_img, threshold_value, 255, THRESH_BINARY);

        // 이진화된 이미지를 벡터로 변환
        vector<float> input(InputSize);
        for (int i = 0; i < binary_img.rows; i++) {
            for (int j = 0; j < binary_img.cols; j++) {
                // 이진화된 이미지의 픽셀 값을 0 또는 1로 변환
                input[i * binary_img.cols + j] = binary_img.at<uchar>(i, j) / 255.0f;
            }
        }

        // 레이블 생성
        vector<float> label(OutputSize, 0);
        label[label_index] = 1;
        dataset.push_back(make_pair(input, label));
        image_count++;
    }

    std::random_device rd_test;
    std::mt19937 g_test(rd_test());
    std::shuffle(dataset.begin(), dataset.end(), g_test);
    return dataset;
}

// 각 클래스별 데이터를 순환적으로 학습 데이터셋에 추가하는 함수
vector<pair<vector<float>, vector<float>>> create_balanced_train_dataset(
    const vector<pair<vector<float>, vector<float>>>& dog_train,
    const vector<pair<vector<float>, vector<float>>>& cat_train,
    const vector<pair<vector<float>, vector<float>>>& hyena_train,
    const vector<pair<vector<float>, vector<float>>>& tiger_train) {

    vector<pair<vector<float>, vector<float>>> train_dataset;
    int max_size = min({dog_train.size(), cat_train.size(), hyena_train.size(), tiger_train.size()});

    // 각 클래스에서 하나씩 순서대로 추가
    for (int i = 0; i < max_size; i++) {
        train_dataset.push_back(dog_train[i]);
        train_dataset.push_back(cat_train[i]);
        train_dataset.push_back(hyena_train[i]);
        train_dataset.push_back(tiger_train[i]);
    }

    return train_dataset;
}

// 결과 텍스트 파일로 저장하는 함수
void log_results_to_file(int epoch, float train_loss, float test_loss,
                         const int* test_correct_counts, const int* test_total_counts, 
                         const string* class_animal, int layers, const vector<int>& nodes_per_layer, bool first_run) {

    // 파일 경로를 동적으로 생성
    stringstream ss;
    ss << "result/Layer" << layers;
    int idx = 0;
    for (int nodes : nodes_per_layer) {
        idx++;
        if(idx==1 || idx == layers){
            continue;
        }
        ss << "_" << nodes;
    }
    ss << "_" << fixed << LearningRate;
    string directory = ss.str();
    string filename = directory + "/result.txt";

    // 디렉토리가 없으면 생성
    fs::create_directories(directory);

    // 파일을 append 모드로 열기
    ofstream log_file(filename, ios::app);
    if (!log_file.is_open()) {
        cerr << "Error opening log file!" << endl;
        return;
    }

    // 처음 로그를 작성할 때만 네트워크 구조와 학습률을 기록
    if (first_run) {
        log_file << "Neural Network Configuration:\n";
        log_file << "Number of Layers: " << layers << "\n";
        log_file << "Nodes per Layer: ";
        for (int nodes : nodes_per_layer) {
            log_file << nodes << " ";
        }
        log_file << "\n";
        log_file << "Learning Rate: " << LearningRate << "\n\n";
    }

    // 에폭 정보와 손실 기록
    log_file << "Epoch " << epoch + 1 
             << " Train_Loss = " << train_loss 
             << ", Test Loss = " << test_loss << "\n";

    // 각 클래스별 정확도 기록
    for (int i = 0; i < OutputSize; i++) {
        float accuracy = (test_total_counts[i] > 0) ? (100.0f * test_correct_counts[i] / test_total_counts[i]) : 0.0f;
        log_file << class_animal[i] << " Accuracy: " << accuracy << "%\n";
    }
    log_file << "\n";

    // 파일 닫기
    log_file.close();
}


int main() {

    // 모델 구조 고정 설정
    int layers = 8;
    vector<int> nodes_per_layer = {65536, 256, 128, 64, 64, 128, 256, 4};
    string class_animal[OutputSize] = {"Dog", "Cat", "Hyena", "Tiger"};

    cout << "Number of layers: " << layers << endl;
    cout << "Nodes per layer: ";
    for (int nodes : nodes_per_layer) {
        cout << nodes << " ";
    }
    cout << endl;

    // 가중치 할당 및 초기화
    float* weights[LayerLimit - 1];
    int r_seed = 1;
    for (int i = 0; i < layers - 1; i++) {
        CUDA_CHECK(cudaMalloc(&weights[i], nodes_per_layer[i] * nodes_per_layer[i + 1] * sizeof(float)));
        initialize_weights<<<(nodes_per_layer[i] * nodes_per_layer[i + 1] + 255) / 256, 256>>>(
            weights[i], nodes_per_layer[i], nodes_per_layer[i + 1], r_seed
        );
        CUDA_CHECK(cudaDeviceSynchronize()); // 가중치 초기화 동기화
    }
    cout << "weights ready" << endl;

    // 각 클래스별로 데이터를 로드
    vector<pair<vector<float>, vector<float>>> dog_images = load_images("dataset/dog/train", 0);
    vector<pair<vector<float>, vector<float>>> cat_images = load_images("dataset/cat/train", 1);
    vector<pair<vector<float>, vector<float>>> hyena_images = load_images("dataset/hyena/train", 2);
    vector<pair<vector<float>, vector<float>>> tiger_images = load_images("dataset/tiger/train", 3);

    cout << "Loaded dog images: " << dog_images.size() << endl;
    cout << "Loaded cat images: " << cat_images.size() << endl;
    cout << "Loaded hyena images: " << hyena_images.size() << endl;
    cout << "Loaded tiger images: " << tiger_images.size() << endl;

    // 각 클래스별로 데이터를 학습 및 테스트 데이터로 분리
    auto split_dataset = [](vector<pair<vector<float>, vector<float>>>& images, float train_ratio = 0.8) {
        int train_size = images.size() * train_ratio;
        vector<pair<vector<float>, vector<float>>> train(images.begin(), images.begin() + train_size);
        vector<pair<vector<float>, vector<float>>> test(images.begin() + train_size, images.end());
        return make_pair(train, test);
    };

    // 각 클래스별로 분할된 데이터셋
    auto [dog_train, dog_test] = split_dataset(dog_images);
    auto [cat_train, cat_test] = split_dataset(cat_images);
    auto [hyena_train, hyena_test] = split_dataset(hyena_images);
    auto [tiger_train, tiger_test] = split_dataset(tiger_images);

    // 학습 데이터와 테스트 데이터를 하나의 벡터로 합침
    vector<pair<vector<float>, vector<float>>> train_dataset = create_balanced_train_dataset(
        dog_train, cat_train, hyena_train, tiger_train
    );
    vector<pair<vector<float>, vector<float>>> test_dataset;

    test_dataset.insert(test_dataset.end(), dog_test.begin(), dog_test.end());
    test_dataset.insert(test_dataset.end(), cat_test.begin(), cat_test.end());
    test_dataset.insert(test_dataset.end(), hyena_test.begin(), hyena_test.end());
    test_dataset.insert(test_dataset.end(), tiger_test.begin(), tiger_test.end());

    std::random_device rd_test;
    std::mt19937 g_test(rd_test());
    std::shuffle(test_dataset.begin(), test_dataset.end(), g_test);

    cout << "Training dataset size: " << train_dataset.size() << endl;
    cout << "Testing dataset size: " << test_dataset.size() << endl;

    // GPU 메모리 할당
    float* d_target;
    float* node_inputs[LayerLimit];
    float* node_errors[LayerLimit];
    float* node_outputs[LayerLimit];

    CUDA_CHECK(cudaMalloc(&d_target, OutputSize * sizeof(float)));

    for (int i = 0; i < layers; i++) {
        CUDA_CHECK(cudaMalloc(&node_errors[i], nodes_per_layer[i] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&node_outputs[i], nodes_per_layer[i] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&node_inputs[i], nodes_per_layer[i] * sizeof(float)));
    }

    bool first_run = true;

    // 학습 루프
    for (int epoch = 0; epoch < 100; epoch++) {
        float total_loss = 0.0f;

        // 학습 데이터셋에 대해 순전파와 역전파 수행
        for (auto& data : train_dataset) {
            vector<float>& input = data.first;
            vector<float>& target = data.second;

            CUDA_CHECK(cudaMemcpy(node_outputs[0], input.data(), InputSize * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_target, target.data(), OutputSize * sizeof(float), cudaMemcpyHostToDevice));

            for (int i = 0; i < layers; i++) {
                CUDA_CHECK(cudaMemset(node_errors[i], 0, nodes_per_layer[i] * sizeof(float)));
            }

            // 순전파
            for (int i = 0; i < layers - 1; i++) {
                bool is_last = (i==layers-2);
                forward_propagation<<<(nodes_per_layer[i + 1] + 255) / 256, 256>>>(
                    node_outputs[i], weights[i], node_inputs[i+1], node_outputs[i+1], nodes_per_layer[i], nodes_per_layer[i + 1], is_last
                );
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            float output[OutputSize];
            CUDA_CHECK(cudaMemcpy(output, node_outputs[layers - 1], OutputSize * sizeof(float), cudaMemcpyDeviceToHost));

            float errors[OutputSize];
            // CUDA_CHECK(cudaMemcpy(errors, node_errors[layers - 1], OutputSize * sizeof(float), cudaMemcpyDeviceToHost));

            total_loss += cross_entropy_loss(output, target.data(), errors, OutputSize);
            
            CUDA_CHECK(cudaMemcpy(node_errors[layers - 1], errors, OutputSize * sizeof(float), cudaMemcpyHostToDevice));

            // 역전파 - 출력층
            backward_propagation_output<<<(OutputSize + 255) / 256, 256>>>(
                node_errors[layers-2], node_errors[layers - 1], 
                node_outputs[layers - 1], d_target, 
                weights[layers - 2], node_outputs[layers - 2], 
                nodes_per_layer[layers - 2]
            );
            CUDA_CHECK(cudaDeviceSynchronize());

            // 역전파 - 은닉층
            for (int i = layers - 2; i >= 1; i--) {
                backward_propagation_hidden<<<(nodes_per_layer[i] + 255) / 256, 256>>>(
                    node_errors[i], node_errors[i + 1], 
                    weights[i - 1], weights[i], 
                    node_outputs[i - 1], node_inputs[i], 
                    nodes_per_layer[i - 1], nodes_per_layer[i], nodes_per_layer[i+1]
                );
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        // 테스트 데이터셋 평가
        float test_loss = 0.0f;
        int test_correct_counts[OutputSize] = {0};
        int test_total_counts[OutputSize] = {0};

        // 테스트 전용 메모리 할당
        float* node_inputs_test[LayerLimit];
        float* node_outputs_test[LayerLimit];
        float* node_errors_test[LayerLimit];

        CUDA_CHECK(cudaMalloc(&node_outputs_test[0], InputSize * sizeof(float)));
        for (int i = 1; i < layers; i++) {
            CUDA_CHECK(cudaMalloc(&node_outputs_test[i], nodes_per_layer[i] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&node_errors_test[i], nodes_per_layer[i] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&node_inputs_test[i], nodes_per_layer[i] * sizeof(float)));
        }

        // 테스트 루프에서 전용 메모리 사용
        for (auto& data : test_dataset) {
            vector<float>& input = data.first;
            vector<float>& target = data.second;

            CUDA_CHECK(cudaMemcpy(node_outputs_test[0], input.data(), InputSize * sizeof(float), cudaMemcpyHostToDevice));

            // 순전파 (테스트 전용 메모리 사용)
            for (int i = 0; i < layers - 1; i++) {
                bool is_last = (i == layers - 2);
                forward_propagation<<<(nodes_per_layer[i + 1] + 255) / 256, 256>>>(
                    node_outputs_test[i], weights[i], node_inputs_test[i + 1], node_outputs_test[i + 1], nodes_per_layer[i], nodes_per_layer[i + 1], is_last
                );
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            float output[OutputSize];
            CUDA_CHECK(cudaMemcpy(output, node_outputs_test[layers - 1], OutputSize * sizeof(float), cudaMemcpyDeviceToHost));

            float errors[OutputSize];
            CUDA_CHECK(cudaMemcpy(errors, node_errors_test[layers - 1], OutputSize * sizeof(float), cudaMemcpyDeviceToHost));

            test_loss += cross_entropy_loss(output, target.data(), errors, OutputSize);

            int predicted_class = max_element(output, output + OutputSize) - output;
            int true_class = max_element(target.begin(), target.end()) - target.begin();
            if (predicted_class == true_class) test_correct_counts[true_class]++;
            test_total_counts[true_class]++;
        }

        // 테스트 전용 메모리 해제
        for (int i = 0; i < layers; i++) {
            cudaFree(node_outputs_test[i]);
            cudaFree(node_errors_test[i]);
            cudaFree(node_inputs_test[i]);
        }

        // 결과 출력
        printf("Epoch %d Train_Loss = %f, Test Loss = %f, Learning Rate = %f\n", epoch + 1, total_loss / train_dataset.size(), test_loss / test_dataset.size(), LearningRate);
        for (int i = 0; i < OutputSize; i++) {
            printf("%s Accuracy: %.2f%%\n", class_animal[i].c_str(), (test_total_counts[i] > 0) ? (100.0f * test_correct_counts[i] / test_total_counts[i]) : 0.0f);
        }
        printf("\n");

        // 로그 파일에 기록
        log_results_to_file(epoch,
                            total_loss / train_dataset.size(), 
                            test_loss / test_dataset.size(),
                            test_correct_counts, 
                            test_total_counts, 
                            class_animal, 
                            layers, 
                            nodes_per_layer, 
                            first_run);

        // 첫 번째 기록 이후로는 네트워크 구조를 기록하지 않음
        first_run = false;
    }

    // 메모리 해제
    cudaFree(d_target);
    for (int i = 0; i < layers - 1; i++) {
        cudaFree(weights[i]);
        cudaFree(node_errors[i]);
        cudaFree(node_inputs[i]);
        cudaFree(node_outputs[i]);
    }

    return 0;
}
