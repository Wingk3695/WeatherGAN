import os
import json
import random
from PIL import Image
import torch
from torchvision import transforms

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


class UnpairedStarDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Args:
            dataset_folder (str): 数据集文件夹路径，如 'path/to/BDD100k'
            split (str): 'train', 'val' 或 'test'
            image_prep (callable or str): 图像预处理方式或transform
            tokenizer: 文本编码tokenizer，调用 tokenizer(text, return_tensors='pt') 等
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.split = split
        self.tokenizer = tokenizer

        self.images_dir = os.path.join(dataset_folder, split)
        assert os.path.isdir(self.images_dir), f"目录不存在: {self.images_dir}"

        # 收集所有图像文件名（假设扩展名为jpg或png）
        all_image_fnames = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.png'))]
        all_image_fnames.sort()

        # 读取所有json描述，建立图像名到描述的映射
        self.captions = {}
        for img_name in all_image_fnames:
            json_path = os.path.join(self.images_dir, os.path.splitext(img_name)[0] + ".json")
            if not os.path.isfile(json_path):
                raise FileNotFoundError(f"缺少对应json: {json_path}")

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            attr = data.get("attributes", {})
            if any(attr.get(k, "").lower() == "undefined" for k in ["weather", "timeofday"]):
                continue
            # 构建自然语言描述
            desc = self._attributes_to_caption(attr)
            self.captions[img_name] = desc

        # 只保留有描述的图片名
        self.image_fnames = list(self.captions.keys())
        self.image_fnames.sort()

        self.transform = build_transform(image_prep)

    def _attributes_to_caption(self, attr):
        # 组合成简洁自然语言描述
        weather = attr.get("weather", "")
        scene = attr.get("scene", "")
        timeofday = attr.get("timeofday", "")
        parts = []
        if weather:
            parts.append(weather)
        if scene:
            parts.append(scene)
        if timeofday:
            parts.append(timeofday)
        caption = " ".join(parts).strip()
        if not caption:
            caption = "unknown scene"
        return caption

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        # 1) 取源图像和描述
        src_img_name = self.image_fnames[idx]
        src_img_path = os.path.join(self.images_dir, src_img_name)
        src_image = Image.open(src_img_path).convert("RGB")
        src_image = self.transform(src_image)
        src_caption = self.captions[src_img_name]

        # 2) 随机采样目标图像，要求描述不与源图像完全相同（避免训练混淆）
        # 这里最多尝试10次，不成功就强制返回不同文件名
        for _ in range(10):
            tgt_idx = random.randint(0, len(self.image_fnames) - 1)
            tgt_img_name = self.image_fnames[tgt_idx]
            tgt_caption = self.captions[tgt_img_name]
            if tgt_caption != src_caption or tgt_img_name != src_img_name:
                break
        else:
            # 万一10次还是一样，强制选用不同文件名
            tgt_idx = (idx + 1) % len(self.image_fnames)
            tgt_img_name = self.image_fnames[tgt_idx]
            tgt_caption = self.captions[tgt_img_name]

        tgt_img_path = os.path.join(self.images_dir, tgt_img_name)
        tgt_image = Image.open(tgt_img_path).convert("RGB")
        tgt_image = self.transform(tgt_image)

        # 3) 文本tokenize
        src_token_ids = self.tokenizer(src_caption, truncation=True, padding="max_length", max_length=77, return_tensors="pt")["input_ids"].squeeze(0)
        tgt_token_ids = self.tokenizer(tgt_caption, truncation=True, padding="max_length", max_length=77, return_tensors="pt")["input_ids"].squeeze(0)

        return {
            "src_image": src_image,
            "src_token_ids": src_token_ids,
            "src_caption": src_caption,
            "tgt_image": tgt_image,
            "tgt_token_ids": tgt_token_ids,
            "tgt_caption": tgt_caption,
        }
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # 假设你用了 huggingface 的 CLIP tokenizer，换成你的tokenizer即可
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    dataset_folder = r"D:\GIT\Datasets\BDD100k\100k"  # 自己改成正确路径
    split = "train"
    image_prep = "resize_256x256"

    dataset = UnpairedStarDataset(dataset_folder, split, image_prep, tokenizer)

    print(f"数据集大小: {len(dataset)}")
    for i in range(5):
        sample = dataset[i]
        print(f"样本{i}:")
        print("  src_caption:", sample["src_caption"])
        print("  tgt_caption:", sample["tgt_caption"])
        print("  src_token_ids shape:", sample["src_token_ids"].shape)
        print("  tgt_token_ids shape:", sample["tgt_token_ids"].shape)
        print("  src_image shape:", sample["src_image"].size)
        print("  tgt_image shape:", sample["tgt_image"].size)
