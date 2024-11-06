from PIL import Image
import numpy as np
import os

# 입력 및 출력 경로 설정
input_dir = "dataset/hyena/converted"
output_dir = "dataset/hyena/augmented_noise"
os.makedirs(output_dir, exist_ok=True)

def add_gaussian_noise(image):
    """가우시안 노이즈 추가"""
    np_img = np.array(image)
    mean = 0
    sigma = 25
    noise = np.random.normal(mean, sigma, np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """소금-후추 노이즈 추가"""
    np_img = np.array(image)
    total_pixels = np_img.size
    salt_pixels = int(salt_prob * total_pixels)
    pepper_pixels = int(pepper_prob * total_pixels)

    # Add salt (white) noise
    for _ in range(salt_pixels):
        x, y = np.random.randint(0, np_img.shape[0]), np.random.randint(0, np_img.shape[1])
        np_img[x, y] = 255

    # Add pepper (black) noise
    for _ in range(pepper_pixels):
        x, y = np.random.randint(0, np_img.shape[0]), np.random.randint(0, np_img.shape[1])
        np_img[x, y] = 0

    return Image.fromarray(np_img)

def add_speckle_noise(image):
    """스피클 노이즈 추가"""
    np_img = np.array(image) / 255.0
    noise = np.random.normal(0, 0.1, np_img.shape)
    noisy_img = np.clip((np_img + np_img * noise) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# 원본 파일 읽기 및 증강
original_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

for i, filename in enumerate(original_files):
    img_path = os.path.join(input_dir, filename)
    
    # 이미지 열기
    with Image.open(img_path) as img:
        img.save(os.path.join(output_dir, f"{i+1}_original.jpg"))  # 원본 저장

        # 가우시안 노이즈 추가
        gaussian_noisy_img = add_gaussian_noise(img)
        gaussian_noisy_img.save(os.path.join(output_dir, f"{i+1}_gaussian.jpg"))

        # 소금-후추 노이즈 추가
        sp_noisy_img = add_salt_pepper_noise(img)
        sp_noisy_img.save(os.path.join(output_dir, f"{i+1}_saltpepper.jpg"))

        # 스피클 노이즈 추가
        speckle_noisy_img = add_speckle_noise(img)
        speckle_noisy_img.save(os.path.join(output_dir, f"{i+1}_speckle.jpg"))

print("노이즈 추가 및 저장 완료!")
