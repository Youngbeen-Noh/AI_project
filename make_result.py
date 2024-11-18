import os
import re
import matplotlib.pyplot as plt

def parse_results(filename):
    """
    주어진 텍스트 파일에서 에폭당 손실 및 클래스별 정확도를 파싱하는 함수.
    """
    epochs = []
    train_loss = []
    test_loss = []
    dog_acc = []
    cat_acc = []
    hyena_acc = []
    tiger_acc = []

    with open(filename, 'r') as file:
        for line in file:
            epoch_match = re.search(r'Epoch (\d+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)
            
            train_loss_match = re.search(r'Train_Loss = ([\d\.]+)', line)
            test_loss_match = re.search(r'Test Loss = ([\d\.]+)', line)
            if train_loss_match:
                train_loss.append(float(train_loss_match.group(1)))
            if test_loss_match:
                test_loss.append(float(test_loss_match.group(1)))

            if 'Dog Accuracy' in line:
                dog_acc_match = re.search(r'Dog Accuracy:\s*([\d\.]+)%', line)
                if dog_acc_match:
                    dog_acc.append(float(dog_acc_match.group(1)))

            if 'Cat Accuracy' in line:
                cat_acc_match = re.search(r'Cat Accuracy:\s*([\d\.]+)%', line)
                if cat_acc_match:
                    cat_acc.append(float(cat_acc_match.group(1)))
            
            if 'Hyena Accuracy' in line:
                hyena_acc_match = re.search(r'Hyena Accuracy:\s*([\d\.]+)%', line)
                if hyena_acc_match:
                    hyena_acc.append(float(hyena_acc_match.group(1)))
            
            if 'Tiger Accuracy' in line:
                tiger_acc_match = re.search(r'Tiger Accuracy:\s*([\d\.]+)%', line)
                if tiger_acc_match:
                    tiger_acc.append(float(tiger_acc_match.group(1)))

    return epochs, train_loss, test_loss, dog_acc, cat_acc, hyena_acc, tiger_acc


def plot_losses(epochs, train_loss, test_loss, folder):
    """
    학습 손실과 테스트 손실을 그래프로 시각화하고 저장하는 함수.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', marker='x')
    plt.title('Training and Test Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0.0, 2.0)  # Y축 범위 설정
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, 'Loss_per_Epoch.png'))
    plt.close()


def plot_accuracies(epochs, dog_acc, cat_acc, hyena_acc, tiger_acc, folder):
    """
    각 클래스별 정확도를 하나의 그래프로 시각화하고 저장하는 함수.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dog_acc, label='Dog Accuracy', marker='o', color='blue')
    plt.plot(epochs, cat_acc, label='Cat Accuracy', marker='x', color='green')
    plt.plot(epochs, hyena_acc, label='Hyena Accuracy', marker='^', color='red')
    plt.plot(epochs, tiger_acc, label='Tiger Accuracy', marker='s', color='purple')
    plt.title('Class Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, 'Class_Accuracy_per_Epoch.png'))
    plt.close()


def process_result_file(folder):
    """
    주어진 폴더에서 result.txt 파일을 읽고 분석 및 그래프를 생성하는 함수.
    """
    filename = os.path.join(folder, 'result.txt')
    if os.path.exists(filename):
        epochs, train_loss, test_loss, dog_acc, cat_acc, hyena_acc, tiger_acc = parse_results(filename)
        if epochs:
            plot_losses(epochs, train_loss, test_loss, folder)
            plot_accuracies(epochs, dog_acc, cat_acc, hyena_acc, tiger_acc, folder)
            print(f"Processed {filename}")
        else:
            print(f"No data found in {filename}")
    else:
        print(f"{filename} does not exist.")


def main():
    base_folder = './result'
    # 'result' 폴더 내의 모든 하위 폴더를 탐색
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            process_result_file(folder_path)


if __name__ == "__main__":
    main()
