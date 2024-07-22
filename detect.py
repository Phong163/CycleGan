import numpy as np
import torch
import cv2
from generator_resnet import Generator
import os
from config import *



def run(input_folder, output_folder):
    files = os.listdir(input_folder)
    image_files = [file for file in files if file.lower().endswith(('.jpg', '.png', 'jpeg'))]

    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found at path: {img_path}")

        # Chuyển ảnh từ BGR sang RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Áp dụng các phép biến đổi
        transformed = transforms(image=img_rgb)
        images = transformed['image'].unsqueeze(0)  # Thêm batch dimension
        images = images.to(device)
        
        with torch.no_grad():
            result = gen(images)
        
        # Chuyển đổi tensor về NumPy array
        result_np = result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        images_np = images.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # Chuyển đổi giá trị pixel về 0-255 và kiểu uint8
        result_np = (result_np * 0.5 + 0.5) * 255
        images_np = (images_np * 0.5 + 0.5) * 255
        result_np = np.clip(result_np, 0, 255).astype(np.uint8)
        images_np = np.clip(images_np, 0, 255).astype(np.uint8)

        # Ghép hai ảnh theo chiều ngang
        combined_image = np.hstack((images_np, result_np))
        
        # Lưu ảnh ghép
        combined_image_path = os.path.join(output_folder, f"combined{i}.png")
        cv2.imwrite(combined_image_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
        print(f'Image saved to {combined_image_path}')

device = torch.device('cpu')
gen = Generator(input_nc=3, output_nc=3, ngf=64, drop=0.5).to(device)
checkpoint_file = r"C:\Users\OS\Desktop\My_project\GAN\CycleGan\weigts\weight1\genz.pth.tar"

# Tải trọng số vào mô hình
print("=> Loading checkpoint")
checkpoint = torch.load(checkpoint_file, map_location=device)
gen.load_state_dict(checkpoint["state_dict"])

input_folder = r"C:\Users\OS\Desktop\My_project\GAN\CycleGan\image"
output_folder = r"C:\Users\OS\Desktop\My_project\GAN\CycleGan\result"
run(input_folder, output_folder)
