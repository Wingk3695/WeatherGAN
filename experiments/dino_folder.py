import os
from PIL import Image
import torch
from dino_struct import DinoStructureLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = DinoStructureLoss()  # 里面已自动用cuda

target_dir = "/your/path/to/target_dir"
output_dir = "/your/path/to/output_dir"

target_files = sorted(os.listdir(target_dir))
output_files = sorted(os.listdir(output_dir))

# 假设文件名一一对应（如 0001.png, 0002.png ...）
inputs = []
outputs = []

for t_name, o_name in zip(target_files, output_files):
    target_img = Image.open(os.path.join(target_dir, t_name)).convert("RGB")
    output_img = Image.open(os.path.join(output_dir, o_name)).convert("RGB")
    input_tensor = loss_fn.preprocess(target_img)
    output_tensor = loss_fn.preprocess(output_img)
    inputs.append(input_tensor)
    outputs.append(output_tensor)

inputs = torch.stack(inputs).to(device)
outputs = torch.stack(outputs).to(device)

loss = loss_fn.calculate_global_ssim_loss(outputs, inputs)
print("DINO Structure Loss:", loss.item())