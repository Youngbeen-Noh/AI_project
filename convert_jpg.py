from PIL import Image
import os

# 원본 디렉터리 경로와 저장할 디렉터리 설정
input_dirs = {
    "dataset/hyena/train": "dataset/hyena/converted",
    "dataset/tiger/train": "dataset/tiger/converted"
}

# 출력 크기
output_size = (256, 256)

# 각 폴더에 대한 처리
for input_dir, output_dir in input_dirs.items():
    # 저장할 폴더가 없다면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            
            # 이미지 열기, 흑백 변환, 리사이즈
            with Image.open(img_path) as img:
                img_gray = img.convert("L").resize(output_size)
                
                # 변환된 이미지를 새로운 경로에 저장
                output_path = os.path.join(output_dir, filename)
                img_gray.save(output_path)

print("이미지 변환 및 저장 완료!")
