import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import json

def build_transform(image_prep):
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    else:
        raise ValueError(f"Unknown image_prep: {image_prep}")
    return T

class PairedMultiDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        super().__init__()
        self.dataset_folder = dataset_folder
        prompts_file = os.path.join(dataset_folder, f"{split}_prompts_with_depth.json")
        with open(prompts_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conditioning_path = item.get("A_path") or item.get("conditioning_image")
        image_path = item.get("B_path") or item.get("image")
        depth_path = item.get("depth_path")  # 新增深度图路径字段
        caption = item["text"]

        if conditioning_path is None or image_path is None:
            raise ValueError(f"路径数据缺失: {item}")

        if not os.path.isabs(conditioning_path):
            conditioning_path = os.path.join(self.dataset_folder, conditioning_path)
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.dataset_folder, image_path)

        conditioning_img = Image.open(conditioning_path).convert("RGB")
        output_img = Image.open(image_path).convert("RGB")

        conditioning_t = self.T(conditioning_img)
        conditioning_t = F.to_tensor(conditioning_t)

        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5]*3, std=[0.5]*3)

        # 处理深度图，允许没有depth_path时返回None或全零tensor
        if depth_path is not None:
            if not os.path.isabs(depth_path):
                depth_path = os.path.join(self.dataset_folder, depth_path)
            depth_img = Image.open(depth_path).convert("L")  # 灰度
            # 转成RGB三通道（复制三次）
            depth_img = Image.merge("RGB", (depth_img, depth_img, depth_img))
            depth_t = self.T(depth_img)
            depth_t = F.to_tensor(depth_t)
            # 归一化到[0,1]，没有归一化到[-1,1]，按需调整
        else:
            # 如果没有depth，返回全零tensor，shape和conditioning_t一样
            depth_t = torch.zeros_like(conditioning_t)

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": conditioning_t,
            "depth_pixel_values": depth_t,
            "caption": caption,
            "input_ids": input_ids,
        }
