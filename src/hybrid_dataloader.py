import os
from glob import glob
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import logging

def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.
    """
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
    return T


class HybridDataset(Dataset):
    def __init__(self, root_dir=None, domainA_paired=None, domainB_paired=None,
                 domainA_unpaired=None, domainB_unpaired=None,
                 caption_a2b=None, caption_b2a=None, transform="resize_256", tokenizer=None):
        """
        初始化 HybridDataset，记录 4 种情况的文件列表，并支持根据模式和方向返回图像对及对应的 caption。

        Args:
            root_dir (str): 数据集根目录，包含四个子文件夹和两个 caption 文件。
            domainA_paired (str): 配对数据中 Domain A 的图像文件夹路径。
            domainB_paired (str): 配对数据中 Domain B 的图像文件夹路径。
            domainA_unpaired (str): 非配对数据中 Domain A 的图像文件夹路径。
            domainB_unpaired (str): 非配对数据中 Domain B 的图像文件夹路径。
            caption_a2b (str): a2b 转换的 caption 文件路径。
            caption_b2a (str): b2a 转换的 caption 文件路径。
            transform (str): 图像预处理方法。
            tokenizer (callable): 文本 tokenizer，用于处理 caption。
        """
        if root_dir:
            domainA_paired = os.path.join(root_dir, "domainA_paired")
            domainB_paired = os.path.join(root_dir, "domainB_paired")
            domainA_unpaired = os.path.join(root_dir, "domainA_unpaired")
            domainB_unpaired = os.path.join(root_dir, "domainB_unpaired")
            caption_a2b = os.path.join(root_dir, "fixed_prompt_a.txt")
            caption_b2a = os.path.join(root_dir, "fixed_prompt_b.txt")
        elif all([domainA_paired, domainB_paired, 
                  domainA_unpaired, domainB_unpaired, caption_a2b, caption_b2a]):
            pass
        else:
            raise ValueError("Either root_dir or all domain and caption paths must be provided.\n" \
            "current params:" \
            "root_dir: {}, " \
            "domainA_paired: {}, " \
            "domainB_paired: {}, " \
            "domainA_unpaired: {}, " \
            "domainB_unpaired: {}, " \
            "caption_a2b: {}, " \
            "caption_b2a: {}".format(root_dir, 
                                     domainA_paired, 
                                     domainB_paired, 
                                     domainA_unpaired, 
                                     domainB_unpaired, 
                                     caption_a2b, 
                                     caption_b2a))
            
        

        # 加载文件路径
        self.domainA_paired = self._load_image_paths(domainA_paired)
        self.domainB_paired = self._load_image_paths(domainB_paired)
        assert len(self.domainA_paired) == len(self.domainB_paired), \
            f"Paired datasets must have the same number of images. " \
            f"Got {len(self.domainA_paired)} and {len(self.domainB_paired)}."
        self.domainA_unpaired = self._load_image_paths(domainA_unpaired)
        self.domainB_unpaired = self._load_image_paths(domainB_unpaired)

        # logging
        logging.info(f"Paired dataset A: {len(self.domainA_paired)} images")
        logging.info(f"Paired dataset B: {len(self.domainB_paired)} images")
        logging.info(f"Unpaired dataset A: {len(self.domainA_unpaired)} images")
        logging.info(f"Unpaired dataset B: {len(self.domainB_unpaired)} images")

        # 加载 captions
        self.caption_a2b = self._load_caption(caption_a2b)
        self.caption_b2a = self._load_caption(caption_b2a)

        # logging
        logging.info(f"Caption A2B: {self.caption_a2b}")
        logging.info(f"Caption B2A: {self.caption_b2a}")

        # Tokenize captions
        self.input_ids_a2b = tokenizer(self.caption_a2b, return_tensors="pt", padding="max_length", truncation=True).input_ids
        self.input_ids_b2a = tokenizer(self.caption_b2a, return_tensors="pt", padding="max_length", truncation=True).input_ids

        # 图像预处理
        self.transform = build_transform(transform)
        logging.info(f"Transform: {transform}")

    def _load_image_paths(self, folder):
        """
        加载文件夹中的所有图像路径。
        """
        if folder is None:
            return []
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob(os.path.join(folder, ext)))
        return image_paths

    def _load_caption(self, file_path):
        """
        加载 caption 文件内容。
        """
        if file_path is None:
            return ""
        with open(file_path, "r") as f:
            return f.read().strip()

    def __len__(self):
        """
        返回数据集的长度，取决于 4 种情况中最大的列表长度。
        """
        return max(len(self.domainA_paired), len(self.domainB_paired), len(self.domainA_unpaired), len(self.domainB_unpaired))

    def __getitem__(self, idx, mode="Paired", direction="a2b"):
        """
        根据模式和方向返回图像对及对应的 caption。

        Args:
            idx (int): 数据索引。
            mode (str): 数据模式，"Paired" 或 "Unpaired"。
            direction (str): 数据方向，"a2b" 或 "b2a"。

        Returns:
            dict: 包含图像对及对应 caption 的字典。
        """
        if mode == "Paired":
            if direction == "a2b":
                imgA_path = self.domainA_paired[idx % len(self.domainA_paired)]
                imgB_path = self.domainB_paired[idx % len(self.domainB_paired)]
                caption = self.caption_a2b
                input_ids = self.input_ids_a2b
            elif direction == "b2a":
                imgA_path = self.domainB_paired[idx % len(self.domainB_paired)]
                imgB_path = self.domainA_paired[idx % len(self.domainA_paired)]
                caption = self.caption_b2a
                input_ids = self.input_ids_b2a
        elif mode == "Unpaired":
            if direction == "a2b":
                imgA_path = self.domainA_unpaired[idx % len(self.domainA_unpaired)]
                imgB_path = random.choice(self.domainB_unpaired)
                caption = self.caption_a2b
                input_ids = self.input_ids_a2b
            elif direction == "b2a":
                imgA_path = self.domainB_unpaired[idx % len(self.domainB_unpaired)]
                imgB_path = random.choice(self.domainA_unpaired)
                caption = self.caption_b2a
                input_ids = self.input_ids_b2a
        else:
            raise ValueError(f"Invalid mode: {mode}. Supported modes are 'Paired' and 'Unpaired'.")

        # 加载图像
        imgA = Image.open(imgA_path).convert("RGB")
        imgB = Image.open(imgB_path).convert("RGB")

        # 应用预处理
        imgA = F.to_tensor(self.transform(imgA))
        imgB = F.to_tensor(self.transform(imgB))
        imgA = F.normalize(imgA, mean=[0.5], std=[0.5])
        imgB = F.normalize(imgB, mean=[0.5], std=[0.5])

        return {
            "pixel_values_src": imgA,
            "pixel_values_tgt": imgB,
            "caption": caption,
            "input_ids": input_ids,
        }
    
if __name__ == "__main__":
    # Example usage
    dataset = HybridDataset(
        root_dir="",
        transform="resize_256",
        tokenizer=None  # Replace with actual tokenizer if needed
    )

    # Get a sample
    sample = dataset[0, "Paired", "a2b"]
    print(sample)